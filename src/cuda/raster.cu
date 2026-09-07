// Fragment stage. Both passes share the bins the binning stage produced and
// differ only in what they write: the colour pass shades and keeps normals for
// SSAO, the shadow pass writes depth alone.
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"
// cuda_barycentric and the whole fragment shader live here now, because the
// deferred path in shade.cu runs exactly the same code.
#include "shading.cuh"

// fill zbuffer without the old CPU malloc+loop+memcpy nonsense
__global__
void zbuffer_fill_kernel(int* zbuffer, int fill_val, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        zbuffer[idx] = fill_val;
}

// shadow pass: same tiling and bins, depth only. keeps the LARGEST depth,
// matching the z-buffer convention (z grows toward the viewer, nearest == max).
__global__
void shadow_raster_kernel(const CudaTriangle* triangles,
                          const int* tile_counts, const int* tile_offsets,
                          const int* tri_indices, int* shadowbuf,
                          int width, int height, int tiles_x)
{
    int tile = blockIdx.y * tiles_x + blockIdx.x;
    int n = tile_counts[tile];
    if (n == 0) return;

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    if (x >= width || y >= height) return;

    const int* bin = tri_indices + tile_offsets[tile];

    for (int k = 0; k < n; k++) {
        const CudaTriangle& tri = triangles[bin[k]];
        if (x < tri.bbox_min_x || x > tri.bbox_max_x ||
            y < tri.bbox_min_y || y > tri.bbox_max_y) continue;

        float v0x = tri.v[0].x / tri.v[0].w, v0y = tri.v[0].y / tri.v[0].w;
        float v1x = tri.v[1].x / tri.v[1].w, v1y = tri.v[1].y / tri.v[1].w;
        float v2x = tri.v[2].x / tri.v[2].w, v2y = tri.v[2].y / tri.v[2].w;
        CudaVec3 bary = cuda_barycentric(v0x,v0y,v1x,v1y,v2x,v2y,(float)x,(float)y);
        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float z = tri.v[0].z*bary.x + tri.v[1].z*bary.y + tri.v[2].z*bary.z;
        float w = tri.v[0].w*bary.x + tri.v[1].w*bary.y + tri.v[2].w*bary.z;
        float depth = z / w;
        if (depth < 0.0f) continue;
        atomicMax(&shadowbuf[y * width + x], __float_as_int(depth));
    }
}

// pass 2: one block per tile, one thread per pixel in that tile. walks only
// the tile's own bin. every thread in the block reads the same triangle, so
// the load broadcasts out of cache rather than thrashing it.
// zbuffer is int-typed so atomicMax can be used: IEEE754 preserves ordering
// when positive floats are reinterpreted as int.
// NOTE the direction. the CPU reference keeps the LARGEST depth, so nearest == max.
// atomicMin here would draw the far surface, and a coverage-only test cannot
// see the difference because the set of lit pixels is identical either way.
__global__
void tiled_raster_kernel(const CudaTriangle* triangles,
                         const int* tile_counts, const int* tile_offsets,
                         const int* tri_indices, const CudaMaterial* materials,
                         float4* framebuffer, int* zbuffer,
                         const int* shadowbuf, float* normalbuf,
                         int width, int height, int tiles_x, int* stats)
{
    int tile = blockIdx.y * tiles_x + blockIdx.x;

    int n = tile_counts[tile];
    if (n == 0) return;               // empty tile, whole block retires

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;

    // colour is stored top-down so the tone map's output can be DMA'd straight
    // into an SDL RGB24 texture with zero per-pixel host work.
    // the zbuffer stays bottom-up (kernel-internal, nobody outside reads it).
    int color_idx = (height - 1 - y) * width + x;

    const int* bin = tri_indices + tile_offsets[tile];

    for (int k = 0; k < n; k++) {
        const CudaTriangle& tri = triangles[bin[k]];

        // the triangle overlaps this tile, but not necessarily this pixel
        if (x < tri.bbox_min_x || x > tri.bbox_max_x ||
            y < tri.bbox_min_y || y > tri.bbox_max_y)
            continue;

        float v0x = tri.v[0].x / tri.v[0].w;
        float v0y = tri.v[0].y / tri.v[0].w;
        float v1x = tri.v[1].x / tri.v[1].w;
        float v1y = tri.v[1].y / tri.v[1].w;
        float v2x = tri.v[2].x / tri.v[2].w;
        float v2y = tri.v[2].y / tri.v[2].w;

        CudaVec3 bary = cuda_barycentric(v0x, v0y, v1x, v1y, v2x, v2y,
                                          (float)x, (float)y);

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float invw0 = 1.0f / tri.v[0].w;
        float invw1 = 1.0f / tri.v[1].w;
        float invw2 = 1.0f / tri.v[2].w;
        float persp_denom = bary.x * invw0 + bary.y * invw1 + bary.z * invw2;
        if (fabsf(persp_denom) < 1e-12f) continue;
        float pw0 = (bary.x * invw0) / persp_denom;
        float pw1 = (bary.y * invw1) / persp_denom;
        float pw2 = (bary.z * invw2) / persp_denom;

        float z = tri.v[0].z * bary.x + tri.v[1].z * bary.y + tri.v[2].z * bary.z;
        float w = tri.v[0].w * bary.x + tri.v[1].w * bary.y + tri.v[2].w * bary.z;
        float depth = z / w;

        if (depth < 0.0f) continue;

        // atomic depth test via int reinterpretation.
        // the color write below isn't atomic with the depth: two tris at the
        // exact same depth could race on color. in practice this is a one-pixel
        // one-frame glitch that you'd never notice. the depth buffer itself
        // stays correct either way.
        int depth_int = __float_as_int(depth);
        int old = atomicMax(&zbuffer[pixel_idx], depth_int);

        if (depth_int >= old) {
            const CudaMaterial& mat = materials[tri.mat];

            // stats[4] counts fragment-shader invocations. It is the whole
            // point of the deferred comparison: forward runs this once per
            // fragment that ever wins the depth test, deferred once per
            // covered pixel, and if those two numbers come out equal then the
            // scene has no overdraw and the benchmark is measuring nothing.
            atomicAdd(&stats[4], 1);

            ShadedFragment f = shade_fragment(tri, mat, x, y, depth,
                                              pw0, pw1, pw2,
                                              shadowbuf, width, height);

            framebuffer[color_idx] = make_float4(f.r * (1.f / 255.f),
                                                 f.g * (1.f / 255.f),
                                                 f.b * (1.f / 255.f), 1.f);

            // eye-space normal for SSAO. indexed like the zbuffer (bottom-up),
            // since that is the space the occlusion pass works in. Unlit and
            // shadow-debug fragments have none to give.
            if (normalbuf && f.has_normal) {
                normalbuf[pixel_idx * 3 + 0] = f.gnx;
                normalbuf[pixel_idx * 3 + 1] = f.gny;
                normalbuf[pixel_idx * 3 + 2] = f.gnz;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Visibility pass: the same walk, but it records who won instead of shading.
// ---------------------------------------------------------------------------
// One 64-bit atomicMax per candidate fragment resolves depth and carries the
// winning triangle's index with it, so the pixel ends up knowing exactly what
// covers it without anything having been shaded yet. shade.cu then runs the
// fragment shader once per pixel.
//
// This also closes a race the forward path documents and tolerates: there, the
// depth write is atomic but the colour write after it is not, so two triangles
// at the same depth can interleave and the colour can come from the loser.
// Here identity and depth move in a single atomic, so that cannot happen.
__global__
void visibility_raster_kernel(const CudaTriangle* triangles,
                              const int* tile_counts, const int* tile_offsets,
                              const int* tri_indices,
                              unsigned long long* visbuffer,
                              int width, int height, int tiles_x)
{
    int tile = blockIdx.y * tiles_x + blockIdx.x;
    int n = tile_counts[tile];
    if (n == 0) return;

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;
    const int* bin = tri_indices + tile_offsets[tile];

    for (int k = 0; k < n; k++) {
        int tri_id = bin[k];
        const CudaTriangle& tri = triangles[tri_id];

        if (x < tri.bbox_min_x || x > tri.bbox_max_x ||
            y < tri.bbox_min_y || y > tri.bbox_max_y) continue;

        float v0x = tri.v[0].x / tri.v[0].w, v0y = tri.v[0].y / tri.v[0].w;
        float v1x = tri.v[1].x / tri.v[1].w, v1y = tri.v[1].y / tri.v[1].w;
        float v2x = tri.v[2].x / tri.v[2].w, v2y = tri.v[2].y / tri.v[2].w;

        CudaVec3 bary = cuda_barycentric(v0x, v0y, v1x, v1y, v2x, v2y,
                                         (float)x, (float)y);
        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float z = tri.v[0].z*bary.x + tri.v[1].z*bary.y + tri.v[2].z*bary.z;
        float w = tri.v[0].w*bary.x + tri.v[1].w*bary.y + tri.v[2].w*bary.z;
        float depth = z / w;
        if (depth < 0.0f) continue;

        atomicMax(&visbuffer[pixel_idx], vis_pack(depth, tri_id));
    }
}

__global__
void visbuffer_fill_kernel(unsigned long long* visbuffer, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) visbuffer[idx] = VIS_EMPTY;
}

void cudaLaunchZbufferFill(int* zbuffer, int fill_val, int count)
{
    int block = 256;
    int grid = (count + block - 1) / block;
    zbuffer_fill_kernel<<<grid, block>>>(zbuffer, fill_val, count);
}

void cudaLaunchTiledRaster(const CudaTriangle* triangles, const int* tile_counts,
                           const int* tile_offsets, const int* tri_indices,
                           const CudaMaterial* materials,
                           float4* framebuffer, int* zbuffer,
                           const int* shadowbuf, float* normalbuf,
                           int width, int height, int tiles_x, int tiles_y,
                           int* stats)
{
    dim3 block(TILE_W, TILE_H);
    dim3 grid(tiles_x, tiles_y);
    tiled_raster_kernel<<<grid, block>>>(triangles, tile_counts, tile_offsets,
                                         tri_indices, materials, framebuffer,
                                         zbuffer, shadowbuf, normalbuf,
                                         width, height, tiles_x, stats);
}

void cudaLaunchShadowRaster(const CudaTriangle* triangles, const int* tile_counts,
                            const int* tile_offsets, const int* tri_indices,
                            int* shadowbuf,
                            int width, int height, int tiles_x, int tiles_y)
{
    dim3 block(TILE_W, TILE_H);
    dim3 grid(tiles_x, tiles_y);
    shadow_raster_kernel<<<grid, block>>>(triangles, tile_counts, tile_offsets,
                                          tri_indices, shadowbuf,
                                          width, height, tiles_x);
}

void cudaLaunchVisibilityRaster(const CudaTriangle* triangles,
                                const int* tile_counts, const int* tile_offsets,
                                const int* tri_indices,
                                unsigned long long* visbuffer,
                                int width, int height, int tiles_x, int tiles_y)
{
    dim3 block(TILE_W, TILE_H);
    dim3 grid(tiles_x, tiles_y);
    visibility_raster_kernel<<<grid, block>>>(triangles, tile_counts,
                                              tile_offsets, tri_indices,
                                              visbuffer, width, height, tiles_x);
}

void cudaLaunchVisbufferFill(unsigned long long* visbuffer, int count)
{
    int block = 256;
    int grid = (count + block - 1) / block;
    visbuffer_fill_kernel<<<grid, block>>>(visbuffer, count);
}
