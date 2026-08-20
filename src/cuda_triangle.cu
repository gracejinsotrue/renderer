// cuda_triangle.cu
// batched CUDA rasterizer. triangles are queued host-side and flushed
// in one kernel launch instead of one launch per triangle.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cub/cub.cuh>

// forward declare rasterizer types
class TGAImage;
class TGAColor;
template<size_t DIM, typename T> struct vec;
typedef vec<3, float> Vec3f;
typedef vec<4, float> Vec4f;

struct CudaVec3 {
    float x, y, z;
    __device__ __host__ CudaVec3() : x(0), y(0), z(0) {}
    __device__ __host__ CudaVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct CudaVec4 {
    float x, y, z, w;
    __device__ __host__ CudaVec4() : x(0), y(0), z(0), w(1) {}
    __device__ __host__ CudaVec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct CudaColor {
    unsigned char r, g, b;
    __device__ __host__ CudaColor() : r(0), g(0), b(0) {}
    __device__ __host__ CudaColor(unsigned char r_, unsigned char g_, unsigned char b_)
        : r(r_), g(g_), b(b_) {}
};

// per-triangle data uploaded to GPU in one shot
struct CudaTriangle {
    CudaVec4 v[3];
    // eye-space normals and uvs per corner, interpolated per fragment
    float nx[3], ny[3], nz[3];
    float u[3], vt[3];
    CudaColor color;
    int mat;              // index into the per-frame material table
    // screen-space bounding box, precomputed on CPU so the kernel can skip early
    int bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y;
};

// one per mesh per frame. the raster kernel rasterizes every mesh in a single
// launch, so per-mesh state has to be reachable from the triangle itself.
struct CudaMaterial {
    cudaTextureObject_t diffuse, nm, spec;
    int has_diffuse, has_nm, has_spec;
    int unlit;            // slot 0: emit the triangle colour verbatim
    int has_shadow;
    float shadow_bias;
    int debug_shadow;     // 1: emit (sz, stored, 0) instead of shading
    // maps a fragment's screen-space position into shadow-map space:
    // light_viewport*proj*modelview * inverse(camera_viewport*proj*modelview)
    float mshadow[16];
    float lx, ly, lz;     // light direction in eye space for this mesh
    float lcr, lcg, lcb;  // light colour in RGB, 0..1
    float lintensity;     // scales the direct term, ambient stays fixed
    // 3x3 of the inverse-transpose. the normal map is object-space, so the
    // sampled normal must be carried to eye space like the vertex normals.
    // using it raw collapses the diffuse term to near zero.
    float mit[9];
};
// starting capacity only. the table is uploaded fresh each flush and grown on
// demand, so there is no cap on how many meshes a frame may draw.
static const int MATERIALS_INITIAL = 64;
static bool g_use_linear_filter = true;

__device__
CudaVec3 cuda_barycentric(float ax, float ay, float bx, float by,
                          float cx, float cy, float px, float py) {
    float s0x = cx - ax;
    float s0y = cy - ay;
    float s1x = bx - ax;
    float s1y = by - ay;
    // must be P - A for this cross product layout. our_gl.cpp writes A - P,
    // but lays its cross product out differently, which cancels the sign.
    float s2x = px - ax;
    float s2y = py - ay;

    float cross_z = s0x * s1y - s0y * s1x;

    if (fabsf(cross_z) < 1e-6f) {
        return CudaVec3(-1, 1, 1);
    }

    float u = (s1y * s2x - s1x * s2y) / cross_z;
    float v = (s0x * s2y - s0y * s2x) / cross_z;

    return CudaVec3(1.0f - u - v, v, u);
}

// Screen tiling. Each triangle is binned into the tiles its bbox touches,
// then each tile rasterizes only its own list, instead of every pixel walking
// every triangle.
#define TILE_W 16
#define TILE_H 16

// fill zbuffer without the old CPU malloc+loop+memcpy nonsense
__global__
void zbuffer_fill_kernel(int* zbuffer, int fill_val, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        zbuffer[idx] = fill_val;
}

// pass 1a: count how many tiles each triangle lands in. counting only, so
// there is no per-tile capacity that could drop geometry.
__global__
void count_kernel(const CudaTriangle* triangles, int num_triangles,
                  int* tile_counts, int tiles_x, int tiles_y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    const CudaTriangle& tri = triangles[t];
    int tx0 = tri.bbox_min_x / TILE_W, tx1 = tri.bbox_max_x / TILE_W;
    int ty0 = tri.bbox_min_y / TILE_H, ty1 = tri.bbox_max_y / TILE_H;
    if (tx0 < 0) tx0 = 0;
    if (ty0 < 0) ty0 = 0;
    if (tx1 >= tiles_x) tx1 = tiles_x - 1;
    if (ty1 >= tiles_y) ty1 = tiles_y - 1;

    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++)
            atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
}

// pass 1b: with an exclusive prefix sum of the counts in hand, every triangle
// writes into its tile's exact slice of one flat index buffer.
__global__
void scatter_kernel(const CudaTriangle* triangles, int num_triangles,
                    const int* tile_offsets, int* tile_cursor, int* tri_indices,
                    int tiles_x, int tiles_y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    const CudaTriangle& tri = triangles[t];
    int tx0 = tri.bbox_min_x / TILE_W, tx1 = tri.bbox_max_x / TILE_W;
    int ty0 = tri.bbox_min_y / TILE_H, ty1 = tri.bbox_max_y / TILE_H;
    if (tx0 < 0) tx0 = 0;
    if (ty0 < 0) ty0 = 0;
    if (tx1 >= tiles_x) tx1 = tiles_x - 1;
    if (ty1 >= tiles_y) ty1 = tiles_y - 1;

    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++) {
            int tile = ty * tiles_x + tx;
            tri_indices[tile_offsets[tile] + atomicAdd(&tile_cursor[tile], 1)] = t;
        }
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
// NOTE the direction. our_gl.cpp keeps the LARGEST depth, so nearest == max.
// atomicMin here would draw the far surface, and a coverage-only test cannot
// see the difference because the set of lit pixels is identical either way.
__global__
void tiled_raster_kernel(const CudaTriangle* triangles,
                         const int* tile_counts, const int* tile_offsets,
                         const int* tri_indices, const CudaMaterial* materials,
                         unsigned char* framebuffer, int* zbuffer,
                         const int* shadowbuf,
                         int width, int height, int tiles_x)
{
    int tile = blockIdx.y * tiles_x + blockIdx.x;

    int n = tile_counts[tile];
    if (n == 0) return;               // empty tile, whole block retires

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;

    // colour is stored top-down and in R,G,B order so the finished frame can be
    // DMA'd straight into an SDL RGB24 texture with zero per-pixel host work.
    // the zbuffer stays bottom-up (kernel-internal, nobody outside reads it).
    int color_idx = ((height - 1 - y) * width + x) * 3;

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

            // the host staging path (cudaRenderTriangle) carries no normals or
            // uvs, so it opts out of shading and its colour passes straight
            // through, which is the contract the tests rely on.
            if (mat.unlit) {
                framebuffer[color_idx + 0] = tri.color.r;
                framebuffer[color_idx + 1] = tri.color.g;
                framebuffer[color_idx + 2] = tri.color.b;
                continue;
            }

            // depth still uses screen-space barycentrics; attributes use
            // perspective-correct weights reconstructed from reciprocal w.
            float nxi = tri.nx[0]*pw0 + tri.nx[1]*pw1 + tri.nx[2]*pw2;
            float nyi = tri.ny[0]*pw0 + tri.ny[1]*pw1 + tri.ny[2]*pw2;
            float nzi = tri.nz[0]*pw0 + tri.nz[1]*pw1 + tri.nz[2]*pw2;
            float nlen = sqrtf(nxi*nxi + nyi*nyi + nzi*nzi);
            if (nlen > 1e-12f) { nxi /= nlen; nyi /= nlen; nzi /= nlen; }
            // everything here survived backface culling, so orient toward camera
            if (nzi < 0.0f) { nxi = -nxi; nyi = -nyi; nzi = -nzi; }

            float uu = tri.u[0]*pw0 + tri.u[1]*pw1 + tri.u[2]*pw2;
            float vv = tri.vt[0]*pw0 + tri.vt[1]*pw1 + tri.vt[2]*pw2;

            // normal map, if present, replaces the interpolated normal. the map
            // stores xyz in BGR order, matching Model::normal(uv) on the CPU.
            if (mat.has_nm) {
                float4 nmc = tex2D<float4>(mat.nm, uu, vv);
                // Model::normal(uv) reads res[2-i] from channel i, i.e. x<-R,
                // y<-G, z<-B; the texture is stored .x=B .y=G .z=R
                float ox = nmc.z * 2.0f - 1.0f;
                float oy = nmc.y * 2.0f - 1.0f;
                float oz = nmc.x * 2.0f - 1.0f;
                float ex = mat.mit[0]*ox + mat.mit[1]*oy + mat.mit[2]*oz;
                float ey = mat.mit[3]*ox + mat.mit[4]*oy + mat.mit[5]*oz;
                float ez = mat.mit[6]*ox + mat.mit[7]*oy + mat.mit[8]*oz;
                float ml = sqrtf(ex*ex + ey*ey + ez*ez);
                if (ml > 1e-12f) {
                    nxi = ex/ml; nyi = ey/ml; nzi = ez/ml;
                    // the mapped normal must still face the viewer; MIT carries
                    // a perspective term that can flip its sense
                    if (nzi < 0.0f) { nxi = -nxi; nyi = -nyi; nzi = -nzi; }
                }
            }

            float diff = nxi*mat.lx + nyi*mat.ly + nzi*mat.lz;
            if (diff < 0.0f) diff = 0.0f;

            // shadow lookup. the fragment's screen-space position goes back
            // through the camera transform and forward through the light's, so
            // (x, y, depth) is exactly what the matrix expects.
            float shadow = 1.0f;
            if (mat.has_shadow) {
                float fx = (float)x, fy = (float)y, fz = depth;
                const float* M = mat.mshadow;
                float sx = M[0]*fx + M[1]*fy + M[2]*fz + M[3];
                float sy = M[4]*fx + M[5]*fy + M[6]*fz + M[7];
                float sz = M[8]*fx + M[9]*fy + M[10]*fz + M[11];
                float sw = M[12]*fx + M[13]*fy + M[14]*fz + M[15];
                if (fabsf(sw) > 1e-12f) { sx /= sw; sy /= sw; sz /= sw; }
                int ix = (int)sx, iy = (int)sy;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    float stored = __int_as_float(shadowbuf[iy * width + ix]);
                    // nearest-to-light is the MAX depth, so a fragment is
                    // lit when it is at least as near as what the light
                    // recorded. bias is a parameter because depth spans 0..255
                    // regardless of world scale.
                    shadow = (sz + mat.shadow_bias >= stored) ? 1.0f : 0.3f;
                    if (mat.debug_shadow) {
                        framebuffer[color_idx + 0] = (unsigned char)fminf(fmaxf(sz,0.f),255.f);
                        framebuffer[color_idx + 1] = (unsigned char)fminf(fmaxf(stored,0.f),255.f);
                        framebuffer[color_idx + 2] = 0;
                        continue;
                    }
                }
            }

            // reflected direction, for the specular lobe
            float rd = 2.0f * (nxi*mat.lx + nyi*mat.ly + nzi*mat.lz);
            float rz = nzi * rd - mat.lz;
            float spec = 0.0f;
            if (mat.has_spec && rz > 0.0f) {
                float p = tex2D<float4>(mat.spec, uu, vv).x * 255.0f;
                if (p < 1.0f) p = 1.0f;
                spec = powf(rz, p);
            }

            float br = tri.color.r, bg = tri.color.g, bb = tri.color.b;
            if (mat.has_diffuse) {
                float4 d = tex2D<float4>(mat.diffuse, uu, vv);
                br = d.z * 255.0f; bg = d.y * 255.0f; bb = d.x * 255.0f;
            }

            // same constants as ShadowMappingShader::fragment on the CPU
            float amb = 20.0f;
            float lit = shadow * mat.lintensity * (0.8f * diff + 0.3f * spec);
            float cr = amb + br * lit * mat.lcr;
            float cg = amb + bg * lit * mat.lcg;
            float cb = amb + bb * lit * mat.lcb;
            framebuffer[color_idx + 0] = (unsigned char)fminf(cr, 255.0f);
            framebuffer[color_idx + 1] = (unsigned char)fminf(cg, 255.0f);
            framebuffer[color_idx + 2] = (unsigned char)fminf(cb, 255.0f);
        }
    }
}

// per-frame triangle capacity. rumi is ~190k faces across its meshes, and
// whole scenes are transformed in one go. ~19 MB at 72 bytes a triangle.
static const int MAX_BATCH = 262144;

// row-major 4x4, passed to the setup kernel by value
struct Mat4 { float m[16]; };

// transforms a whole mesh and does setup (cull + bbox) on the GPU, appending
// survivors straight into the shared triangle buffer.
__global__
void mesh_setup_kernel(const float* verts, const int* faces, int nfaces,
                       const float* cnorms, const float* cuvs,
                       Mat4 mvp, Mat4 clip, Mat4 mit, int mat_id,
                       CudaColor color,
                       CudaTriangle* out, int* out_count, int max_out,
                       int width, int height, int* stats)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nfaces) return;
    atomicAdd(&stats[3], 1);                       // submitted

    CudaVec4 v[3], cv[3];
    float sx[3], sy[3];
    float mx[3], my[3], mz[3];                     // model-space, for the normal
    for (int k = 0; k < 3; k++) {
        int vi = faces[f * 3 + k];
        float px = verts[vi * 3 + 0];
        float py = verts[vi * 3 + 1];
        float pz = verts[vi * 3 + 2];
        mx[k] = px; my[k] = py; mz[k] = pz;
        v[k].x = mvp.m[0]  * px + mvp.m[1]  * py + mvp.m[2]  * pz + mvp.m[3];
        v[k].y = mvp.m[4]  * px + mvp.m[5]  * py + mvp.m[6]  * pz + mvp.m[7];
        v[k].z = mvp.m[8]  * px + mvp.m[9]  * py + mvp.m[10] * pz + mvp.m[11];
        v[k].w = mvp.m[12] * px + mvp.m[13] * py + mvp.m[14] * pz + mvp.m[15];
        cv[k].x = clip.m[0]  * px + clip.m[1]  * py + clip.m[2]  * pz + clip.m[3];
        cv[k].y = clip.m[4]  * px + clip.m[5]  * py + clip.m[6]  * pz + clip.m[7];
        cv[k].z = clip.m[8]  * px + clip.m[9]  * py + clip.m[10] * pz + clip.m[11];
        cv[k].w = clip.m[12] * px + clip.m[13] * py + clip.m[14] * pz + clip.m[15];
        sx[k] = v[k].x / v[k].w;
        sy[k] = v[k].y / v[k].w;
    }

    bool outside = true;
    for (int k = 0; k < 3; k++) if (cv[k].w > 0.0f) outside = false;
    if (outside) { atomicAdd(&stats[1], 1); return; }
    outside = true;
    for (int k = 0; k < 3; k++) if (cv[k].x >= -cv[k].w) outside = false;
    if (outside) { atomicAdd(&stats[1], 1); return; }
    outside = true;
    for (int k = 0; k < 3; k++) if (cv[k].x <= cv[k].w) outside = false;
    if (outside) { atomicAdd(&stats[1], 1); return; }
    outside = true;
    for (int k = 0; k < 3; k++) if (cv[k].y >= -cv[k].w) outside = false;
    if (outside) { atomicAdd(&stats[1], 1); return; }
    outside = true;
    for (int k = 0; k < 3; k++) if (cv[k].y <= cv[k].w) outside = false;
    if (outside) { atomicAdd(&stats[1], 1); return; }
    // same cull test and sign convention as the host path and our_gl.cpp
    float facing = (sx[1] - sx[0]) * (sy[2] - sy[0])
                 - (sy[1] - sy[0]) * (sx[2] - sx[0]);
    if (facing >= 0.0f) { atomicAdd(&stats[0], 1); return; }

    int bx0 = max(0,          (int)floorf(fminf(fminf(sx[0], sx[1]), sx[2])));
    int bx1 = min(width  - 1, (int)ceilf (fmaxf(fmaxf(sx[0], sx[1]), sx[2])));
    int by0 = max(0,          (int)floorf(fminf(fminf(sy[0], sy[1]), sy[2])));
    int by1 = min(height - 1, (int)ceilf (fmaxf(fmaxf(sy[0], sy[1]), sy[2])));
    if (bx0 > bx1 || by0 > by1) { atomicAdd(&stats[1], 1); return; }

    int slot = atomicAdd(out_count, 1);
    if (slot >= max_out) { atomicAdd(&stats[2], 1); return; }

    CudaTriangle& t = out[slot];
    t.v[0] = v[0]; t.v[1] = v[1]; t.v[2] = v[2];
    t.color = color;
    t.mat = mat_id;

    // corner attributes travel with the triangle, since shading is
    // per-fragment. normals go to eye space via the inverse-transpose, using
    // only the 3x3 block, so the 4th component is implicitly 0.
    for (int k = 0; k < 3; k++) {
        float snx, sny, snz;
        if (cnorms) {
            snx = cnorms[(f * 3 + k) * 3 + 0];
            sny = cnorms[(f * 3 + k) * 3 + 1];
            snz = cnorms[(f * 3 + k) * 3 + 2];
        } else {
            // no vertex normals in the file: fall back to the face normal
            float e1x = mx[1] - mx[0], e1y = my[1] - my[0], e1z = mz[1] - mz[0];
            float e2x = mx[2] - mx[0], e2y = my[2] - my[0], e2z = mz[2] - mz[0];
            snx = e1y * e2z - e1z * e2y;
            sny = e1z * e2x - e1x * e2z;
            snz = e1x * e2y - e1y * e2x;
        }
        float ex = mit.m[0] * snx + mit.m[1] * sny + mit.m[2]  * snz;
        float ey = mit.m[4] * snx + mit.m[5] * sny + mit.m[6]  * snz;
        float ez = mit.m[8] * snx + mit.m[9] * sny + mit.m[10] * snz;
        float l = sqrtf(ex * ex + ey * ey + ez * ez);
        if (l > 1e-12f) { ex /= l; ey /= l; ez /= l; }
        t.nx[k] = ex; t.ny[k] = ey; t.nz[k] = ez;
        t.u[k]  = cuvs ? cuvs[(f * 3 + k) * 2 + 0] : 0.0f;
        t.vt[k] = cuvs ? cuvs[(f * 3 + k) * 2 + 1] : 0.0f;
    }
    t.bbox_min_x = bx0; t.bbox_max_x = bx1;
    t.bbox_min_y = by0; t.bbox_max_y = by1;
}

// geometry that lives on the device across frames
struct DeviceMesh {
    float* d_verts;
    int*   d_faces;
    float* d_norms;    // per-corner, nfaces*3*3, may be NULL
    float* d_uvs;      // per-corner, nfaces*3*2, may be NULL
    cudaArray_t  tex_arr[3];      // diffuse, normal map, specular
    cudaTextureObject_t tex[3];
    bool has_tex[3];
    int nverts, nfaces;
    bool alive;
};

// uploads one TGAImage-style buffer (BGR/BGRA/greyscale, bottom-up already
// flipped by Model::load_texture) as an RGBA8 texture.
static bool uploadTexture(const unsigned char* px, int w, int h, int bpp,
                          cudaArray_t* arr_out, cudaTextureObject_t* tex_out)
{
    if (!px || w <= 0 || h <= 0) return false;
    std::vector<unsigned char> rgba((size_t)w * h * 4);
    for (int i = 0; i < w * h; i++) {
        unsigned char b = px[(size_t)i * bpp + 0];
        unsigned char g = (bpp > 1) ? px[(size_t)i * bpp + 1] : b;
        unsigned char r = (bpp > 2) ? px[(size_t)i * bpp + 2] : b;
        rgba[(size_t)i * 4 + 0] = b;   // keep BGRA order; the shader indexes
        rgba[(size_t)i * 4 + 1] = g;   // .x=B .y=G .z=R to match TGAColor
        rgba[(size_t)i * 4 + 2] = r;
        rgba[(size_t)i * 4 + 3] = 255;
    }
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    if (cudaMallocArray(arr_out, &ch, w, h) != cudaSuccess) return false;
    cudaMemcpy2DToArray(*arr_out, 0, 0, rgba.data(), (size_t)w * 4,
                        (size_t)w * 4, h, cudaMemcpyHostToDevice);

    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = *arr_out;

    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    // matches the CPU path's bilinear sampler, so the two rasterizers agree.
    td.filterMode = g_use_linear_filter ? cudaFilterModeLinear : cudaFilterModePoint;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    return cudaCreateTextureObject(tex_out, &rd, &td, NULL) == cudaSuccess;
}

class CudaTriangleRasterizer {
private:
    CudaTriangle* d_triangles;
    unsigned char* d_framebuffer;
    int* d_zbuffer;
    int* d_tile_counts;    // one per tile, reset each flush
    int* d_tile_offsets;   // exclusive prefix sum of the counts
    int* d_tile_cursor;    // write cursor per tile during scatter
    int* d_tri_indices;    // flat, exactly sum(counts) long. grown on demand.
    size_t idx_capacity;   // entries currently allocated in d_tri_indices
    void*  d_scan_temp;    // cub scratch
    size_t scan_temp_bytes;
    int    stat_bin_entries;  // total (triangle, tile) pairs last flush
    int*   d_tri_count;       // triangles appended by mesh_setup_kernel
    int*   d_stats;           // [culled_back, culled_off, dropped, submitted]
    CudaMaterial* d_materials;   // one slot per mesh drawn this frame
    size_t mat_capacity;         // slots currently allocated in d_materials
    int*   d_shadowbuf;          // depth from the light's point of view
    // staged host-side and uploaded in one memcpy per flush. slot 0 is always
    // the neutral pass-through material the host staging path points at.
    std::vector<CudaMaterial> h_materials;
    int width, height;
    int tiles_x, tiles_y, num_tiles;

    // per-stage GPU timing. nsys can't get a GPU timeline through WSL2 and
    // ncu needs a driver permission change, so the kernels time themselves.
    cudaEvent_t ev_start, ev_upload, ev_bin, ev_raster;
    bool timing_ready;
    bool initialized;

    // host-side staging buffer: triangles accumulate here until flush
    std::vector<CudaTriangle> h_batch;

    // per-frame counters, reset in clear(). the kernel walks every queued
    // triangle for every pixel, so "kept" is what actually drives frame cost.
    int stat_submitted, stat_culled_back, stat_culled_offscreen;

public:
    CudaTriangleRasterizer(int w, int h)
        : width(w), height(h), initialized(false),
          stat_submitted(0), stat_culled_back(0), stat_culled_offscreen(0) {
        tiles_x = (w + TILE_W - 1) / TILE_W;
        tiles_y = (h + TILE_H - 1) / TILE_H;
        num_tiles = tiles_x * tiles_y;

        cudaError_t err;

        err = cudaMalloc(&d_triangles, MAX_BATCH * sizeof(CudaTriangle));
        if (err != cudaSuccess) {
            printf("CUDA malloc triangles failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaMalloc(&d_framebuffer, width * height * 3);
        if (err != cudaSuccess) {
            printf("CUDA malloc framebuffer failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles);
            return;
        }

        err = cudaMalloc(&d_zbuffer, width * height * sizeof(int));
        if (err != cudaSuccess) {
            printf("CUDA malloc zbuffer failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles);
            cudaFree(d_framebuffer);
            return;
        }

        err = cudaMalloc(&d_tile_counts, num_tiles * sizeof(int));
        if (err != cudaSuccess) {
            printf("CUDA malloc tile counts failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles); cudaFree(d_framebuffer); cudaFree(d_zbuffer);
            return;
        }

        cudaMalloc(&d_tile_offsets, num_tiles * sizeof(int));
        cudaMalloc(&d_tile_cursor,  num_tiles * sizeof(int));

        // 8 tile-entries per triangle to start, grown on demand.
        idx_capacity = (size_t)MAX_BATCH * 8;
        cudaMalloc(&d_tri_indices, idx_capacity * sizeof(int));

        cudaMalloc(&d_tri_count, sizeof(int));
        cudaMemset(d_tri_count, 0, sizeof(int));
        cudaMalloc(&d_stats, 4 * sizeof(int));
        cudaMemset(d_stats, 0, 4 * sizeof(int));
        mat_capacity = MATERIALS_INITIAL;
        cudaMalloc(&d_materials, mat_capacity * sizeof(CudaMaterial));
        cudaMalloc(&d_shadowbuf, (size_t)w * h * sizeof(int));
        cudaMemset(d_shadowbuf, 0, (size_t)w * h * sizeof(int));
        resetMaterials();

        scan_temp_bytes = 0; d_scan_temp = NULL;
        cub::DeviceScan::ExclusiveSum(NULL, scan_temp_bytes,
                                      d_tile_counts, d_tile_offsets, num_tiles);
        cudaMalloc(&d_scan_temp, scan_temp_bytes);
        stat_bin_entries = 0;

        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_upload);
        cudaEventCreate(&ev_bin);
        cudaEventCreate(&ev_raster);
        timing_ready = false;

        h_batch.reserve(4096);
        initialized = true;
        printf("CUDA rasterizer initialized: %dx%d, %dx%d tiles of %dx%d\n",
               width, height, tiles_x, tiles_y, TILE_W, TILE_H);
    }

    ~CudaTriangleRasterizer() {
        if (initialized) {
            cudaFree(d_triangles);
            cudaFree(d_framebuffer);
            cudaFree(d_zbuffer);
            cudaFree(d_tile_counts);
            cudaFree(d_tile_offsets);
            cudaFree(d_tile_cursor);
            cudaFree(d_tri_indices);
            cudaFree(d_scan_temp);
            cudaFree(d_tri_count);
            cudaFree(d_stats);
            cudaFree(d_materials);
            cudaFree(d_shadowbuf);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_upload);
            cudaEventDestroy(ev_bin);
            cudaEventDestroy(ev_raster);
        }
    }

    bool isInitialized() const { return initialized; }

    void clear() {
        if (!initialized) return;

        cudaMemset(d_framebuffer, 0, width * height * 3);

        // zbuffer stores ints (float-as-int). nearest is the MAXIMUM depth here,
        // so the "empty" value is 0 (+0.0f) and any visible fragment beats it.
        union { float f; int i; } far_val;
        far_val.f = 0.0f;

        int pixel_count = width * height;
        int block = 256;
        int grid = (pixel_count + block - 1) / block;
        zbuffer_fill_kernel<<<grid, block>>>(d_zbuffer, far_val.i, pixel_count);

        h_batch.clear();
        stat_submitted = stat_culled_back = stat_culled_offscreen = 0;
        cudaMemset(d_tri_count, 0, sizeof(int));
        cudaMemset(d_stats, 0, 4 * sizeof(int));
        // zero the table so slot 0 is a valid neutral material: the host
        // staging path points every triangle at it, and reading an
        // uninitialized cudaTextureObject_t would fault.
        cudaMemset(d_shadowbuf, 0, (size_t)width * height * sizeof(int));
        resetMaterials();
    }

    // drops every per-mesh material, leaving slot 0 as the neutral
    // pass-through cudaRenderTriangle points at. safe after any flush: the
    // triangles referencing the old slots have been consumed.
    void resetMaterials() {
        h_materials.clear();
        CudaMaterial neutral;
        memset(&neutral, 0, sizeof(neutral));
        neutral.unlit = 1;
        h_materials.push_back(neutral);
    }

    // milliseconds for the most recent flush. syncs on the last event only.
    void getKernelTimings(float* upload_ms, float* bin_ms, float* raster_ms) {
        if (upload_ms) *upload_ms = 0.f;
        if (bin_ms)    *bin_ms    = 0.f;
        if (raster_ms) *raster_ms = 0.f;
        if (!initialized || !timing_ready) return;
        cudaEventSynchronize(ev_raster);
        if (upload_ms) cudaEventElapsedTime(upload_ms, ev_start,  ev_upload);
        if (bin_ms)    cudaEventElapsedTime(bin_ms,    ev_upload, ev_bin);
        if (raster_ms) cudaEventElapsedTime(raster_ms, ev_bin,    ev_raster);
    }

    void getStats(int* submitted, int* culled_back, int* culled_offscreen,
                  int* bin_overflow) {
        int ds[4] = {0,0,0,0};
        if (initialized)
            cudaMemcpy(ds, d_stats, 4 * sizeof(int), cudaMemcpyDeviceToHost);
        if (submitted)        *submitted        = stat_submitted        + ds[3];
        if (culled_back)      *culled_back      = stat_culled_back      + ds[0];
        if (culled_offscreen) *culled_offscreen = stat_culled_offscreen + ds[1];
        // not an overflow count: bins are sized exactly. this is the total
        // number of (triangle, tile) pairs, i.e. how far binning fans out.
        if (bin_overflow) *bin_overflow = stat_bin_entries;
    }

    // just queue on CPU, no GPU work yet
    void submitTriangle(float v0x, float v0y, float v0z, float v0w,
                        float v1x, float v1y, float v1z, float v1w,
                        float v2x, float v2y, float v2z, float v2w,
                        unsigned char r, unsigned char g, unsigned char b) {
        if (!initialized) return;
        stat_submitted++;

        CudaTriangle tri;
        tri.v[0] = CudaVec4(v0x, v0y, v0z, v0w);
        tri.v[1] = CudaVec4(v1x, v1y, v1z, v1w);
        tri.v[2] = CudaVec4(v2x, v2y, v2z, v2w);
        tri.color = CudaColor(r, g, b);

        // compute screen-space bbox from perspective-divided coords
        float sx0 = v0x / v0w, sy0 = v0y / v0w;
        float sx1 = v1x / v1w, sy1 = v1y / v1w;
        float sx2 = v2x / v2w, sy2 = v2y / v2w;

        // backface cull, same test and sign convention as our_gl.cpp: the
        // sign of the screen-space edge cross product says which way the
        // triangle faces.
        float facing = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0);
        if (facing >= 0.0f) {
            stat_culled_back++;
            return;
        }

        float fmin_x = fminf(fminf(sx0, sx1), sx2);
        float fmax_x = fmaxf(fmaxf(sx0, sx1), sx2);
        float fmin_y = fminf(fminf(sy0, sy1), sy2);
        float fmax_y = fmaxf(fmaxf(sy0, sy1), sy2);

        tri.bbox_min_x = std::max(0, (int)floorf(fmin_x));
        tri.bbox_max_x = std::min(width - 1, (int)ceilf(fmax_x));
        tri.bbox_min_y = std::max(0, (int)floorf(fmin_y));
        tri.bbox_max_y = std::min(height - 1, (int)ceilf(fmax_y));

        // completely offscreen, don't even queue it
        if (tri.bbox_min_x > tri.bbox_max_x || tri.bbox_min_y > tri.bbox_max_y) {
            stat_culled_offscreen++;
            return;
        }

        // the host staging path has no material of its own; slot 0 is reserved
        // as a neutral one so cudaRenderTriangle keeps working for the tests
        tri.mat = 0;
        for (int k = 0; k < 3; k++) {
            tri.nx[k] = 0.f; tri.ny[k] = 0.f; tri.nz[k] = 1.f;
            tri.u[k] = 0.f;  tri.vt[k] = 0.f;
        }
        h_batch.push_back(tri);

        if ((int)h_batch.size() >= MAX_BATCH)
            flush();
    }

    // upload queued triangles and rasterize in one kernel launch
    // shadow_pass: bin exactly the same way, then write depth only.
    void flush(bool shadow_pass = false) {
        if (!initialized) return;

        cudaEventRecord(ev_start);

        // mesh_setup_kernel appends straight into d_triangles, so the device
        // owns the count now. the host staging path (still used by the tests
        // and cudaRenderTriangle) appends after whatever the meshes wrote.
        int num_tris = 0;
        cudaMemcpy(&num_tris, d_tri_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (!h_batch.empty()) {
            int n = (int)h_batch.size();
            if (num_tris + n > MAX_BATCH) n = MAX_BATCH - num_tris;
            if (n > 0) {
                cudaMemcpy(d_triangles + num_tris, h_batch.data(),
                           n * sizeof(CudaTriangle), cudaMemcpyHostToDevice);
                num_tris += n;
                cudaMemcpy(d_tri_count, &num_tris, sizeof(int),
                           cudaMemcpyHostToDevice);
            }
            h_batch.clear();
        }

        // one upload per flush for the whole material table, grown if this
        // frame drew more meshes than any previous one
        if (h_materials.size() > mat_capacity) {
            if (d_materials) cudaFree(d_materials);
            mat_capacity = h_materials.size() + h_materials.size() / 2;
            cudaError_t me = cudaMalloc(&d_materials,
                                        mat_capacity * sizeof(CudaMaterial));
            if (me != cudaSuccess) {
                printf("CUDA material table alloc failed (%zu slots): %s\n",
                       mat_capacity, cudaGetErrorString(me));
                d_materials = NULL; mat_capacity = 0;
                h_batch.clear(); resetMaterials();
                return;
            }
        }
        cudaMemcpy(d_materials, h_materials.data(),
                   h_materials.size() * sizeof(CudaMaterial),
                   cudaMemcpyHostToDevice);

        cudaEventRecord(ev_upload);

        if (num_tris > MAX_BATCH) num_tris = MAX_BATCH;   // setup kernel overran
        if (num_tris == 0) { timing_ready = false; resetMaterials(); return; }

        // pass 1a: count (triangle, tile) pairs. nothing is written yet, so
        // there is no capacity to exceed.
        cudaMemset(d_tile_counts, 0, num_tiles * sizeof(int));

        int binBlock = 256;
        int binGrid = (num_tris + binBlock - 1) / binBlock;
        count_kernel<<<binGrid, binBlock>>>(d_triangles, num_tris,
                                            d_tile_counts, tiles_x, tiles_y);

        // pass 1b: exclusive prefix sum gives each tile its exact slice
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
                                      d_tile_counts, d_tile_offsets, num_tiles);

        // total entries = last offset + last count. one 8-byte readback per
        // flush; it goes away once geometry lives on the device permanently.
        int last_off = 0, last_cnt = 0;
        cudaMemcpy(&last_off, d_tile_offsets + num_tiles - 1, sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_cnt, d_tile_counts  + num_tiles - 1, sizeof(int),
                   cudaMemcpyDeviceToHost);
        size_t total = (size_t)last_off + (size_t)last_cnt;
        stat_bin_entries = (int)total;

        if (total > idx_capacity) {
            cudaFree(d_tri_indices);
            idx_capacity = total + total / 2;      // grow with headroom
            cudaError_t ge = cudaMalloc(&d_tri_indices, idx_capacity * sizeof(int));
            if (ge != cudaSuccess) {
                printf("CUDA tile index buffer alloc failed (%zu entries): %s\n",
                       idx_capacity, cudaGetErrorString(ge));
                d_tri_indices = NULL; idx_capacity = 0;
                h_batch.clear();
                return;
            }
        }

        cudaMemset(d_tile_cursor, 0, num_tiles * sizeof(int));
        scatter_kernel<<<binGrid, binBlock>>>(d_triangles, num_tris,
                                              d_tile_offsets, d_tile_cursor,
                                              d_tri_indices, tiles_x, tiles_y);

        cudaEventRecord(ev_bin);

        // pass 2: one block per tile, one thread per pixel in it
        dim3 blockSize(TILE_W, TILE_H);
        dim3 gridSize(tiles_x, tiles_y);

        if (shadow_pass) {
            shadow_raster_kernel<<<gridSize, blockSize>>>(
                d_triangles, d_tile_counts, d_tile_offsets, d_tri_indices,
                d_shadowbuf, width, height, tiles_x
            );
        } else {
            tiled_raster_kernel<<<gridSize, blockSize>>>(
                d_triangles, d_tile_counts, d_tile_offsets, d_tri_indices,
                d_materials, d_framebuffer, d_zbuffer, d_shadowbuf,
                width, height, tiles_x
            );
        }

        cudaEventRecord(ev_raster);
        timing_ready = true;

        // consumed: the next flush accumulates from zero. the material ids
        // baked into those triangles die with them, so the table resets too
        // and the shadow pass does not eat the colour pass's slots.
        cudaMemset(d_tri_count, 0, sizeof(int));
        resetMaterials();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        }

        h_batch.clear();
    }

    // DMA the finished frame straight into a locked SDL texture. the kernel
    // already wrote it top-down in R,G,B, so no conversion is needed and one
    // 2D memcpy replaces ~2M per-pixel host operations a frame.
    // dst_pitch is SDL's row stride, which may be wider than width * 3.
    unsigned char* deviceFramebuffer(int* w, int* h) {
        if (w) *w = width;
        if (h) *h = height;
        if (!initialized) return NULL;
        flush();
        return d_framebuffer;
    }

    void blitToTexture(void* dst, int dst_pitch) {
        if (!initialized || !dst) return;

        flush();

        cudaError_t err = cudaMemcpy2D(dst, (size_t)dst_pitch,
                                       d_framebuffer, (size_t)width * 3,
                                       (size_t)width * 3, (size_t)height,
                                       cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA blit failed: %s\n", cudaGetErrorString(err));
        }
    }

    // slow path, only when something host-side still needs to composite onto
    // the frame (vertex overlay, ray trace blend) or we're writing a TGA.
    // host_framebuffer is width * height * 3, top-down R,G,B.
    void copyToCPU(unsigned char* host_framebuffer) {
        if (!initialized) return;

        flush();

        cudaMemcpy(host_framebuffer, d_framebuffer, width * height * 3,
                   cudaMemcpyDeviceToHost);
    }

    // ---- persistent device geometry -------------------------------------
    std::vector<DeviceMesh>& meshes() { static std::vector<DeviceMesh> m; return m; }

    int createMesh(const float* verts, int nverts, const int* faces, int nfaces,
                   const float* cnorms, const float* cuvs) {
        if (!initialized) return -1;
        DeviceMesh dm;
        memset(&dm, 0, sizeof(dm));
        dm.nverts = nverts; dm.nfaces = nfaces; dm.alive = true;
        if (cudaMalloc(&dm.d_verts, (size_t)nverts * 3 * sizeof(float)) != cudaSuccess)
            return -1;
        if (cudaMalloc(&dm.d_faces, (size_t)nfaces * 3 * sizeof(int)) != cudaSuccess) {
            cudaFree(dm.d_verts); return -1;
        }
        cudaMemcpy(dm.d_verts, verts, (size_t)nverts * 3 * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dm.d_faces, faces, (size_t)nfaces * 3 * sizeof(int),
                   cudaMemcpyHostToDevice);

        // per-corner normals and uvs, stored unindexed: avoids separate
        // vt/vn index arrays and matches how the OBJ addresses them
        if (cnorms) {
            cudaMalloc(&dm.d_norms, (size_t)nfaces * 9 * sizeof(float));
            cudaMemcpy(dm.d_norms, cnorms, (size_t)nfaces * 9 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        if (cuvs) {
            cudaMalloc(&dm.d_uvs, (size_t)nfaces * 6 * sizeof(float));
            cudaMemcpy(dm.d_uvs, cuvs, (size_t)nfaces * 6 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        meshes().push_back(dm);
        return (int)meshes().size() - 1;
    }

    // called only when the mesh changes (sculpting, blend shapes). a static
    // model is uploaded once and never touched again.
    void updateMeshVerts(int h, const float* verts, int nverts) {
        if (h < 0 || h >= (int)meshes().size()) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive || nverts != dm.nverts) return;
        cudaMemcpy(dm.d_verts, verts, (size_t)nverts * 3 * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void destroyMesh(int h) {
        if (h < 0 || h >= (int)meshes().size()) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive) return;
        cudaFree(dm.d_verts); cudaFree(dm.d_faces);
        if (dm.d_norms) cudaFree(dm.d_norms);
        if (dm.d_uvs)   cudaFree(dm.d_uvs);
        for (int i = 0; i < 3; i++) {
            if (dm.has_tex[i]) {
                cudaDestroyTextureObject(dm.tex[i]);
                cudaFreeArray(dm.tex_arr[i]);
            }
        }
        dm.alive = false;
    }

    void setMeshTexture(int h, int slot, const unsigned char* px,
                        int w, int hgt, int bpp) {
        if (h < 0 || h >= (int)meshes().size() || slot < 0 || slot > 2) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive || dm.has_tex[slot]) return;
        dm.has_tex[slot] = uploadTexture(px, w, hgt, bpp,
                                         &dm.tex_arr[slot], &dm.tex[slot]);
    }

    void renderShadowPass() { flush(true); }

    void shadowStats(int* nonzero, float* mn, float* mx) {
        if (nonzero) *nonzero = 0;
        if (mn) *mn = 0.f;
        if (mx) *mx = 0.f;
        if (!initialized) return;
        std::vector<int> h((size_t)width * height);
        cudaMemcpy(h.data(), d_shadowbuf, h.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        int nz = 0; float lo = 1e30f, hi = -1e30f;
        for (size_t i = 0; i < h.size(); i++) {
            if (h[i] == 0) continue;
            nz++;
            union { int i; float f; } c; c.i = h[i];
            if (c.f < lo) lo = c.f;
            if (c.f > hi) hi = c.f;
        }
        if (nonzero) *nonzero = nz;
        if (nz) { if (mn) *mn = lo; if (mx) *mx = hi; }
    }

    void drawMesh(int h, const float* mvp16, const float* clip16,
                  const float* mit16,
                  const float* light3, const float* light_rgb3, float light_intensity,
                  const float* mshadow16, float shadow_bias,
                  unsigned char r, unsigned char g, unsigned char b) {
        if (!initialized || h < 0 || h >= (int)meshes().size()) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive || dm.nfaces == 0) return;

        // the raster kernel draws every mesh in one launch, so per-mesh state
        // (textures, the eye-space light) goes into a table the triangle indexes
        CudaMaterial m;
        memset(&m, 0, sizeof(m));
        for (int i = 0; i < 3; i++) {
            if (dm.has_tex[i]) {
                if (i == 0) { m.diffuse = dm.tex[0]; m.has_diffuse = 1; }
                if (i == 1) { m.nm      = dm.tex[1]; m.has_nm      = 1; }
                if (i == 2) { m.spec    = dm.tex[2]; m.has_spec    = 1; }
            }
        }
        m.lx = light3[0]; m.ly = light3[1]; m.lz = light3[2];
        m.lcr = light_rgb3 ? fmaxf(light_rgb3[0], 0.0f) : 1.0f;
        m.lcg = light_rgb3 ? fmaxf(light_rgb3[1], 0.0f) : 1.0f;
        m.lcb = light_rgb3 ? fmaxf(light_rgb3[2], 0.0f) : 1.0f;
        m.lintensity = fmaxf(light_intensity, 0.0f);
        if (mshadow16) {
            m.has_shadow = 1;
            m.shadow_bias = shadow_bias;
            m.debug_shadow = (shadow_bias < 0.0f) ? 1 : 0;   // negative bias = debug
            for (int i = 0; i < 16; i++) m.mshadow[i] = mshadow16[i];
        }
        for (int r2 = 0; r2 < 3; r2++)
            for (int c2 = 0; c2 < 3; c2++)
                m.mit[r2 * 3 + c2] = mit16[r2 * 4 + c2];
        // staged, not uploaded: flush() sends the whole table in one memcpy
        h_materials.push_back(m);
        int mat_id = (int)h_materials.size() - 1;

        Mat4 mvp, clip, mit;
        for (int i = 0; i < 16; i++) mvp.m[i] = mvp16[i];
        for (int i = 0; i < 16; i++) clip.m[i] = clip16[i];
        for (int i = 0; i < 16; i++) mit.m[i] = mit16[i];

        int block = 256;
        int grid = (dm.nfaces + block - 1) / block;
        mesh_setup_kernel<<<grid, block>>>(dm.d_verts, dm.d_faces, dm.nfaces,
                                           dm.d_norms, dm.d_uvs,
                                           mvp, clip, mit, mat_id,
                                           CudaColor(r, g, b),
                                           d_triangles, d_tri_count, MAX_BATCH,
                                           width, height, d_stats);
    }

    void synchronize() {
        if (initialized) {
            flush();
            cudaDeviceSynchronize();
        }
    }
};

static CudaTriangleRasterizer* g_cuda_rasterizer = nullptr;

#include "geometry.h"
#include "tgaimage.h"

// C interface, matching the signatures Engine.cpp calls
extern "C" {
    bool initCudaRasterizer(int width, int height) {
        if (g_cuda_rasterizer) delete g_cuda_rasterizer;

        g_cuda_rasterizer = new CudaTriangleRasterizer(width, height);
        return g_cuda_rasterizer->isInitialized();
    }

    void cleanupCudaRasterizer() {
        if (g_cuda_rasterizer) {
            delete g_cuda_rasterizer;
            g_cuda_rasterizer = nullptr;
        }
    }

    void cudaSetLinearTextureFiltering(int enabled) {
        g_use_linear_filter = enabled != 0;
    }

    void cudaClearBuffers() {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->clear();
        }
    }

    // called per-triangle from Engine.cpp. queues only, no GPU work yet.
    void cudaRenderTriangle(const Vec4f& v0, const Vec4f& v1, const Vec4f& v2,
                            const TGAColor& tga_color) {
        if (g_cuda_rasterizer) {
            TGAColor& color = const_cast<TGAColor&>(tga_color);

            g_cuda_rasterizer->submitTriangle(
                v0[0], v0[1], v0[2], v0[3],
                v1[0], v1[1], v1[2], v1[3],
                v2[0], v2[1], v2[2], v2[3],
                color.bgra[2], color.bgra[1], color.bgra[0]
            );
        }
    }

    // ---- persistent geometry: upload once, transform on the GPU each frame
    int cudaCreateMesh(const float* verts, int nverts, const int* faces, int nfaces,
                       const float* corner_normals, const float* corner_uvs) {
        return g_cuda_rasterizer
             ? g_cuda_rasterizer->createMesh(verts, nverts, faces, nfaces,
                                             corner_normals, corner_uvs) : -1;
    }
    // slot: 0 diffuse, 1 normal map, 2 specular. px is a TGAImage buffer.
    void cudaSetMeshTexture(int handle, int slot, const unsigned char* px,
                            int w, int h, int bpp) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->setMeshTexture(handle, slot, px, w, h, bpp);
    }
    void cudaUpdateMeshVerts(int handle, const float* verts, int nverts) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->updateMeshVerts(handle, verts, nverts);
    }
    void cudaDestroyMesh(int handle) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->destroyMesh(handle);
    }
    // mshadow16 may be NULL for an unshadowed draw
    void cudaDrawMesh(int handle, const float* mvp16, const float* clip16,
                      const float* mit16,
                      const float* light3, const float* light_rgb3,
                      float light_intensity, const float* mshadow16,
                      float shadow_bias,
                      unsigned char r, unsigned char g, unsigned char b) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->drawMesh(handle, mvp16, clip16, mit16, light3,
                                        light_rgb3, light_intensity,
                                        mshadow16, shadow_bias, r, g, b);
    }
    void cudaGetShadowStats(int* nonzero, float* mn, float* mx) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->shadowStats(nonzero, mn, mx);
    }
    // rasterizes everything queued so far into the shadow depth buffer
    void cudaRenderShadowPass() {
        if (g_cuda_rasterizer) g_cuda_rasterizer->renderShadowPass();
    }

    void cudaGetKernelTimings(float* upload_ms, float* bin_ms, float* raster_ms) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->getKernelTimings(upload_ms, bin_ms, raster_ms);
    }

    void cudaGetRasterStats(int* submitted, int* culled_back, int* culled_offscreen,
                            int* bin_overflow) {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->getStats(submitted, culled_back, culled_offscreen,
                                        bin_overflow);
        }
    }

    // The finished frame, still on the device. Pending kernels are flushed
    // first so anything that composites into it lands on top of a complete
    // raster frame. Both modules launch on the default stream, so a kernel
    // queued after this call is ordered after the raster work.
    unsigned char* cudaGetDeviceFramebuffer(int* w, int* h) {
        if (!g_cuda_rasterizer) {
            if (w) *w = 0;
            if (h) *h = 0;
            return NULL;
        }
        return g_cuda_rasterizer->deviceFramebuffer(w, h);
    }

    void cudaBlitToTexture(void* dst, int dst_pitch) {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->blitToTexture(dst, dst_pitch);
        }
    }

    // pull the frame back into the host TGAImage so host-side code can
    // composite onto it. TGAImage is bottom-up and stores B,G,R while the
    // device buffer is top-down R,G,B, hence the row flip.
    void cudaCopyResults(TGAImage& framebuffer) {
        if (!g_cuda_rasterizer) return;

        int width = framebuffer.get_width();
        int height = framebuffer.get_height();

        unsigned char* h_framebuffer = new unsigned char[width * height * 3];
        g_cuda_rasterizer->copyToCPU(h_framebuffer);

        for (int y = 0; y < height; y++) {
            const unsigned char* row = h_framebuffer + (size_t)(height - 1 - y) * width * 3;
            for (int x = 0; x < width; x++) {
                TGAColor color(row[x * 3 + 0], row[x * 3 + 1], row[x * 3 + 2]);
                framebuffer.set(x, y, color);
            }
        }

        delete[] h_framebuffer;
    }
}
