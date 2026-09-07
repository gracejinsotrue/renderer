// Deferred shading: one thread per pixel, one shade per pixel.
//
// The forward raster shades inside its depth loop, so a pixel covered by n
// triangles can run the full fragment shader up to n times and keep only the
// last one to win. Cost there scales with depth complexity, not with the
// number of pixels, and the bins are unsorted so nothing bounds how often it
// happens.
//
// Here the visibility pass has already decided who covers each pixel. This
// kernel reads that answer, recomputes the barycentrics for the winner, and
// shades exactly once. Shading cost becomes a function of screen area alone.
//
// Barycentrics are recomputed rather than stored: it is a handful of flops
// against 8 more bytes per pixel of bandwidth, and the visibility buffer is
// already read once per pixel either way.
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"
#include "shading.cuh"

__global__
void deferred_shade_kernel(const CudaTriangle* triangles,
                           const CudaMaterial* materials,
                           const unsigned long long* visbuffer,
                           float4* framebuffer, int* zbuffer,
                           const int* shadowbuf, float* normalbuf,
                           int width, int height, int* stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;
    unsigned long long v = visbuffer[pixel_idx];

    // Nothing covered this pixel. The background is already in the frame from
    // clear(), so leaving it alone is the correct output.
    if (!vis_hit(v)) return;

    int tri_id = vis_tri(v);
    float depth = vis_depth(v);
    const CudaTriangle& tri = triangles[tri_id];

    // SSAO reads depth from the zbuffer, and the visibility pass never wrote
    // one. Publishing it here keeps that stage identical between the two
    // paths rather than teaching it a second buffer layout.
    zbuffer[pixel_idx] = __float_as_int(depth);

    float v0x = tri.v[0].x / tri.v[0].w, v0y = tri.v[0].y / tri.v[0].w;
    float v1x = tri.v[1].x / tri.v[1].w, v1y = tri.v[1].y / tri.v[1].w;
    float v2x = tri.v[2].x / tri.v[2].w, v2y = tri.v[2].y / tri.v[2].w;

    CudaVec3 bary = cuda_barycentric(v0x, v0y, v1x, v1y, v2x, v2y,
                                     (float)x, (float)y);

    float invw0 = 1.0f / tri.v[0].w;
    float invw1 = 1.0f / tri.v[1].w;
    float invw2 = 1.0f / tri.v[2].w;
    float persp_denom = bary.x * invw0 + bary.y * invw1 + bary.z * invw2;
    if (fabsf(persp_denom) < 1e-12f) return;
    float pw0 = (bary.x * invw0) / persp_denom;
    float pw1 = (bary.y * invw1) / persp_denom;
    float pw2 = (bary.z * invw2) / persp_denom;

    const CudaMaterial& mat = materials[tri.mat];

    // counted the same way the forward path counts it, see raster.cu
    atomicAdd(&stats[4], 1);

    ShadedFragment f = shade_fragment(tri, mat, x, y, depth,
                                      pw0, pw1, pw2, shadowbuf, width, height);

    // the frame is stored top-down; everything above is bottom-up
    int color_idx = (height - 1 - y) * width + x;
    // Linear radiance, the same space the environment backdrop is written
    // in, so the tone map at the end sees one scale rather than two.
    framebuffer[color_idx] = make_float4(f.r, f.g, f.b, 1.f);

    if (normalbuf && f.has_normal) {
        normalbuf[pixel_idx * 3 + 0] = f.gnx;
        normalbuf[pixel_idx * 3 + 1] = f.gny;
        normalbuf[pixel_idx * 3 + 2] = f.gnz;
    }
}

void cudaLaunchDeferredShade(const CudaTriangle* triangles,
                             const CudaMaterial* materials,
                             const unsigned long long* visbuffer,
                             float4* framebuffer, int* zbuffer,
                             const int* shadowbuf, float* normalbuf,
                             int width, int height, int* stats)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    deferred_shade_kernel<<<grid, block>>>(triangles, materials, visbuffer,
                                           framebuffer, zbuffer, shadowbuf,
                                           normalbuf, width, height, stats);
}
