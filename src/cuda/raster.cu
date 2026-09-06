// Fragment stage. Both passes share the bins the binning stage produced and
// differ only in what they write: the colour pass shades and keeps normals for
// SSAO, the shadow pass writes depth alone.
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"

__device__
CudaVec3 cuda_barycentric(float ax, float ay, float bx, float by,
                          float cx, float cy, float px, float py) {
    float s0x = cx - ax;
    float s0y = cy - ay;
    float s1x = bx - ax;
    float s1y = by - ay;
    // must be P - A for this cross product layout. the CPU reference writes A - P,
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
                         unsigned char* framebuffer, int* zbuffer,
                         const int* shadowbuf, float* normalbuf,
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

            // Keep the interpolated normal for SSAO before the normal map
            // gets a chance to replace it. The mapped normal carries pore and
            // wrinkle detail, which is what you want for lighting and exactly
            // what you do not want for orienting an occlusion hemisphere: it
            // tilts the hemisphere into the surface and every sample then
            // reads as occluded.
            float gnx = nxi, gny = nyi, gnz = nzi;
            // NOT reoriented toward the camera. an interpolated normal near a
            // silhouette legitimately points away, and forcing nz >= 0 flips
            // the sign of n.l discontinuously wherever nz crosses zero, which
            // shows up as hard-edged bands across a curved surface. the CPU
            // shader does not do it either; max(0, n.l) below handles it.

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
                    // used as-is, not reoriented toward the camera. MIT is
                    // built from ModelView alone, which is affine, so there is
                    // no perspective term to correct for, and a world-space
                    // normal map routinely produces normals facing away.
                    nxi = ex/ml; nyi = ey/ml; nzi = ez/ml;
                }
            }

            float diff = nxi*mat.lx + nyi*mat.ly + nzi*mat.lz;
            if (diff < 0.0f) diff = 0.0f;

            // geometric normal against the light, kept signed and unmapped.
            // the shadow bias below scales with it, and the normal map's
            // high-frequency detail would only make that bias noisy.
            float gdotl = gnx*mat.lx + gny*mat.ly + gnz*mat.lz;

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
                    // How fast this surface recedes from the light across one
                    // shadow texel. Seen edge-on it changes depth a lot, and a
                    // flat bias reads that slope as an occluder, which is what
                    // put dark patches over every curved surface: clothing over
                    // a body sits well inside a constant bias big enough to
                    // cover the glancing case. Scaling by the slope separates
                    // "this surface tilts away from the light" from "something
                    // is actually between it and the light".
                    float slope = sqrtf(fmaxf(0.0f, 1.0f - gdotl * gdotl))
                                / fmaxf(fabsf(gdotl), 0.15f);
                    float bias_eff = mat.shadow_bias * (1.0f + slope);

                    // nearest-to-light is the MAX depth, so a fragment is
                    // lit when it is at least as near as what the light
                    // recorded. bias is a parameter because depth spans 0..255
                    // regardless of world scale.
                    shadow = (sz + bias_eff >= stored) ? 1.0f : 0.3f;
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

            // tests/test_shaded.cpp checks these constants against an
            // independently derived CPU reference
            float amb = 20.0f;
            float lit = shadow * mat.lintensity * (0.8f * diff + 0.3f * spec);
            float cr = amb + br * lit * mat.lcr;
            float cg = amb + bg * lit * mat.lcg;
            float cb = amb + bb * lit * mat.lcb;
            framebuffer[color_idx + 0] = (unsigned char)fminf(cr, 255.0f);
            framebuffer[color_idx + 1] = (unsigned char)fminf(cg, 255.0f);
            framebuffer[color_idx + 2] = (unsigned char)fminf(cb, 255.0f);

            // eye-space normal for SSAO. indexed like the zbuffer (bottom-up),
            // since that is the space the occlusion pass works in.
            if (normalbuf) {
                normalbuf[pixel_idx * 3 + 0] = gnx;
                normalbuf[pixel_idx * 3 + 1] = gny;
                normalbuf[pixel_idx * 3 + 2] = gnz;
            }
        }
    }
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
                           unsigned char* framebuffer, int* zbuffer,
                           const int* shadowbuf, float* normalbuf,
                           int width, int height, int tiles_x, int tiles_y)
{
    dim3 block(TILE_W, TILE_H);
    dim3 grid(tiles_x, tiles_y);
    tiled_raster_kernel<<<grid, block>>>(triangles, tile_counts, tile_offsets,
                                         tri_indices, materials, framebuffer,
                                         zbuffer, shadowbuf, normalbuf,
                                         width, height, tiles_x);
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
