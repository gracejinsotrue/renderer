// The fragment shading math, in one place because two stages now run it.
//
// The forward path (tiled_raster_kernel) shades inside the depth loop, so it
// pays for every fragment that ever won the depth test even if something
// nearer overwrites it a moment later. The deferred path records the winning
// triangle per pixel and shades once, afterwards. They are meant to produce
// the same image, so they call the same function rather than each keeping a
// copy that can drift.
//
// That is deliberate here and would be wrong for a test: the independent
// implementation these are both checked against is the CPU rasterizer in
// tests/reference_raster.cpp. Two shaders that agree because they are the same
// code prove nothing on their own; they are the thing under test, not the
// reference.
#pragma once

#include <cmath>

#include "common.cuh"

__device__ inline
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

struct ShadedFragment {
    float r, g, b;          // 0..255, already clamped
    float gnx, gny, gnz;    // geometric (unmapped) eye-space normal, for SSAO
    // Unlit and shadow-debug fragments have no meaningful normal to hand the
    // occlusion pass, and the forward path skipped the normal write for both.
    bool has_normal;
};

// Everything the old inner loop did once the depth test was won: material
// lookup, perspective-correct attribute interpolation, normal mapping, the
// shadow lookup with its slope-scaled bias, the specular lobe, and the final
// composite.
//
// pw0..pw2 are the perspective-correct barycentric weights; depth is the
// screen-space depth the winner recorded. shadowbuf may be read but is never
// written here.
__device__ inline
ShadedFragment shade_fragment(const CudaTriangle& tri, const CudaMaterial& mat,
                              int x, int y, float depth,
                              float pw0, float pw1, float pw2,
                              const int* shadowbuf, int width, int height)
{
    ShadedFragment out;
    out.has_normal = false;
    out.gnx = 0.f; out.gny = 0.f; out.gnz = 0.f;

    // the host staging path (cudaRenderTriangle) carries no normals or
    // uvs, so it opts out of shading and its colour passes straight
    // through, which is the contract the tests rely on.
    if (mat.unlit) {
        out.r = tri.color.r; out.g = tri.color.g; out.b = tri.color.b;
        return out;
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
                out.r = fminf(fmaxf(sz, 0.f), 255.f);
                out.g = fminf(fmaxf(stored, 0.f), 255.f);
                out.b = 0.f;
                return out;
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
    out.r = fminf(amb + br * lit * mat.lcr, 255.0f);
    out.g = fminf(amb + bg * lit * mat.lcg, 255.0f);
    out.b = fminf(amb + bb * lit * mat.lcb, 255.0f);

    out.gnx = gnx; out.gny = gny; out.gnz = gnz;
    out.has_normal = true;
    return out;
}

// ---------------------------------------------------------------------------
// Visibility buffer packing
// ---------------------------------------------------------------------------
// One 64-bit word per pixel: depth in the high half, triangle index in the
// low half. A single atomicMax then resolves depth and carries the winner's
// identity with it, which is what makes deferred shading correct -- the
// forward path's separate depth-then-colour writes are not atomic together,
// and two triangles at the same depth can race on colour.
//
// Depth is a non-negative float (the raster rejects depth < 0), and IEEE754
// orders non-negative floats identically to their bit patterns read as
// integers, so comparing the packed words compares depth first.
//
// The stored index is tri + 1, so that the all-zero word means "nothing here"
// and cannot be confused with a real hit. Without the bias, triangle 0 sitting
// at depth 0.0 -- the far plane, which is a legal depth -- packs to exactly
// the empty value and the pixel silently reads as background.
#define VIS_EMPTY 0ULL

__device__ inline
unsigned long long vis_pack(float depth, int tri_index)
{
    unsigned int d = __float_as_uint(depth);
    return ((unsigned long long)d << 32) | (unsigned int)(tri_index + 1);
}

__device__ inline
bool vis_hit(unsigned long long v)
{
    return (v & 0xFFFFFFFFULL) != 0ULL;
}

__device__ inline
float vis_depth(unsigned long long v)
{
    return __uint_as_float((unsigned int)(v >> 32));
}

__device__ inline
int vis_tri(unsigned long long v)
{
    return (int)(unsigned int)(v & 0xFFFFFFFFULL) - 1;
}
