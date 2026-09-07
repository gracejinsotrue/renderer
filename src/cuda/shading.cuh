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

#include "brdf.cuh"
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
    // Linear radiance leaving the surface toward the eye. 1.0 is the white
    // point -- a white Lambertian surface facing a unit light -- and values
    // above it are legal and expected: a sun in an environment map is worth
    // tens of these. Clamping is the tone map's job, at the end of the frame.
    float r, g, b;
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
        // Straight through, only rescaled: these bytes are a colour the caller
        // chose, not a measurement of light, so no transfer curve is undone
        // here and none is applied on the way out (the tests that use this
        // path run with the tone map off).
        out.r = tri.color.r * (1.f / 255.f);
        out.g = tri.color.g * (1.f / 255.f);
        out.b = tri.color.b * (1.f / 255.f);
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
                // Depths, not light: 0..255 rescaled to the buffer's range so
                // the passthrough path writes them back unchanged.
                out.r = fminf(fmaxf(sz, 0.f), 255.f) * (1.f / 255.f);
                out.g = fminf(fmaxf(stored, 0.f), 255.f) * (1.f / 255.f);
                out.b = 0.f;
                return out;
            }
        }
    }

    // The view direction, eye space. Taken as +Z for every fragment: exact
    // under an orthographic view, and off by half the field of view at the
    // corners of a perspective one. Correcting it needs the fragment's
    // eye-space position, which neither path carries -- the visibility buffer
    // stores a triangle and a depth, not a position. The Phong lobe this
    // replaced made the same assumption.
    float ndotv = nzi;
    // Half vector between L and V=(0,0,1).
    float hx = mat.lx, hy = mat.ly, hz = mat.lz + 1.0f;
    float hl = sqrtf(hx*hx + hy*hy + hz*hz);
    if (hl > 1e-12f) { hx /= hl; hy /= hl; hz /= hl; }
    float ndoth = nxi*hx + nyi*hy + nzi*hz;
    float vdoth = hz;

    if (ndotv < 0.0f) ndotv = 0.0f;
    if (ndoth < 0.0f) ndoth = 0.0f;
    if (vdoth < 0.0f) vdoth = 0.0f;

    // Roughness. A specular map, where there is one, holds a Phong exponent;
    // p and GGX alpha describe the same lobe width at alpha^2 = 2/(p+2)
    // (Walter et al. 2007, eq. 42), so the maps that ship with the older
    // models keep meaning what they meant. The material scalar covers
    // everything without one.
    float rough = mat.roughness;
    if (mat.has_spec) {
        float p = tex2D<float4>(mat.spec, uu, vv).x * 255.0f;
        if (p < 1.0f) p = 1.0f;
        rough = sqrtf(sqrtf(2.0f / (p + 2.0f)));
    }
    rough = brdf_clamp_roughness(rough);

    // Albedo, decoded out of sRGB. A texture is authored to look right on a
    // display, so its bytes are already through a transfer curve; multiplying
    // them by light without undoing it lights the encoding rather than the
    // surface, and darkens midtones by about a factor of two once the tone map
    // puts the curve back. Vertex colours are treated the same way, since they
    // are picked by eye against the same displays.
    float ar = tri.color.r * (1.f / 255.f);
    float ag = tri.color.g * (1.f / 255.f);
    float ab = tri.color.b * (1.f / 255.f);
    if (mat.has_diffuse) {
        float4 d = tex2D<float4>(mat.diffuse, uu, vv);
        ar = d.z; ag = d.y; ab = d.x;
    }
    ar = srgb_to_linear(ar);
    ag = srgb_to_linear(ag);
    ab = srgb_to_linear(ab);

    // Reflectance at normal incidence. Dielectrics reflect a colourless 4%
    // and keep their albedo for the diffuse lobe; metals reflect their albedo
    // and have no diffuse lobe at all. Metallic interpolates because a texel
    // can straddle the boundary, not because anything is physically in
    // between.
    float f0r = 0.04f + (ar - 0.04f) * mat.metallic;
    float f0g = 0.04f + (ag - 0.04f) * mat.metallic;
    float f0b = 0.04f + (ab - 0.04f) * mat.metallic;

    // Direct light. lintensity is the radiance a white Lambertian surface
    // facing the light would return, so the irradiance it stands for is that
    // times pi -- which is the factor the Lambert BRDF's 1/pi then takes back
    // out. Written this way so the existing default intensity and exposure
    // still mean what they did.
    float el = shadow * mat.lintensity * BRDF_PI;

    float sr = brdf_specular(ndotv, diff, ndoth, vdoth, rough, f0r);
    float sg = brdf_specular(ndotv, diff, ndoth, vdoth, rough, f0g);
    float sb = brdf_specular(ndotv, diff, ndoth, vdoth, rough, f0b);

    // What Fresnel reflects is not available to refract, and a metal refracts
    // nothing regardless.
    float kdr = (1.0f - brdf_F_schlick(f0r, vdoth)) * (1.0f - mat.metallic);
    float kdg = (1.0f - brdf_F_schlick(f0g, vdoth)) * (1.0f - mat.metallic);
    float kdb = (1.0f - brdf_F_schlick(f0b, vdoth)) * (1.0f - mat.metallic);

    const float INV_PI = 1.0f / BRDF_PI;
    out.r = (kdr * ar * INV_PI + sr) * diff * el * mat.lcr;
    out.g = (kdg * ag * INV_PI + sg) * diff * el * mat.lcg;
    out.b = (kdb * ab * INV_PI + sb) * diff * el * mat.lcb;

    // Ambient, as irradiance over pi -- the same quantity the irradiance map
    // stores, which is what lets the two cases share one expression. No
    // environment is a uniform one: constant radiance in every direction
    // convolves to E/pi equal to that same constant, so the fallback is a
    // number rather than a second code path.
    //
    // AMBIENT_L is dim on purpose. It exists so an unlit side is not pure
    // black, and anything brighter starts competing with the light.
    const float AMBIENT_L = 0.03f;
    float er = AMBIENT_L, eg = AMBIENT_L, eb = AMBIENT_L;
    // What a mirror would see. Same fallback logic: reflecting a uniform
    // environment gives back the constant, so this starts equal to the
    // irradiance and is only replaced when there is a map to reflect.
    float pr = AMBIENT_L, pg = AMBIENT_L, pb = AMBIENT_L;

    if (mat.has_irradiance || mat.has_prefiltered) {
        // The mapped normal is used, not the geometric one: unlike the SSAO
        // hemisphere, a texture lookup has nothing to be tilted into.
        const float* M = mat.e2w;
        float wx = M[0]*nxi + M[1]*nyi + M[2]*nzi;
        float wy = M[3]*nxi + M[4]*nyi + M[5]*nzi;
        float wz = M[6]*nxi + M[7]*nyi + M[8]*nzi;
        float wl = sqrtf(wx*wx + wy*wy + wz*wz);
        if (wl > 1e-12f) { wx /= wl; wy /= wl; wz /= wl; }

        if (mat.has_irradiance) {
            float iu, iv;
            equirect_uv(wx, wy, wz, &iu, &iv);
            float4 E = tex2D<float4>(mat.irradiance, iu, iv);
            er = E.x * mat.ibl_intensity;
            eg = E.y * mat.ibl_intensity;
            eb = E.z * mat.ibl_intensity;
        }

        if (mat.has_prefiltered) {
            // Reflect the view about the normal, in eye space, then carry the
            // result to world space. Reflecting after the rotation would work
            // equally well; doing it here keeps V's (0,0,1) explicit.
            float rx = 2.0f * ndotv * nxi;
            float ry = 2.0f * ndotv * nyi;
            float rz = 2.0f * ndotv * nzi - 1.0f;
            float wrx = M[0]*rx + M[1]*ry + M[2]*rz;
            float wry = M[3]*rx + M[4]*ry + M[5]*rz;
            float wrz = M[6]*rx + M[7]*ry + M[8]*rz;
            float rl = sqrtf(wrx*wrx + wry*wry + wrz*wrz);
            if (rl > 1e-12f) { wrx /= rl; wry /= rl; wrz /= rl; }

            float su, sv;
            equirect_uv(wrx, wry, wrz, &su, &sv);

            // Levels are separate maps at separate sizes, not a mip chain, so
            // the blend between the two bracketing roughnesses is done here.
            float lod = rough * (float)(IBL_SPEC_LEVELS - 1);
            int l0 = (int)lod;
            if (l0 < 0) l0 = 0;
            if (l0 > IBL_SPEC_LEVELS - 2) l0 = IBL_SPEC_LEVELS - 2;
            float t = lod - (float)l0;
            if (t < 0.f) t = 0.f;
            if (t > 1.f) t = 1.f;

            float4 c0 = tex2D<float4>(mat.prefiltered[l0], su, sv);
            float4 c1 = tex2D<float4>(mat.prefiltered[l0 + 1], su, sv);
            pr = (c0.x + (c1.x - c0.x) * t) * mat.ibl_intensity;
            pg = (c0.y + (c1.y - c0.y) * t) * mat.ibl_intensity;
            pb = (c0.z + (c1.z - c0.z) * t) * mat.ibl_intensity;
        }
    }

    // The split sum's second factor: the fraction of a white environment the
    // specular lobe returns. It is linear in F0, so one table of a scale and a
    // bias covers every material.
    float far_r, far_g, far_b;
    if (mat.has_brdf_lut) {
        float4 ab_lut = tex2D<float4>(mat.brdf_lut, ndotv, rough);
        far_r = f0r * ab_lut.x + ab_lut.y;
        far_g = f0g * ab_lut.x + ab_lut.y;
        far_b = f0b * ab_lut.x + ab_lut.y;
    } else {
        // No table: Fresnel alone, which over-reflects at grazing angles. The
        // specular ambient is dropped rather than guessed, so what this
        // fallback is really for is keeping the diffuse complement below
        // meaningful. n.v stands in for v.h, since an ambient lookup has no
        // single half vector.
        far_r = brdf_F_schlick_roughness(f0r, ndotv, rough);
        far_g = brdf_F_schlick_roughness(f0g, ndotv, rough);
        far_b = brdf_F_schlick_roughness(f0b, ndotv, rough);
    }

    // The diffuse ambient takes exactly what the specular did not. Deriving it
    // from its own Fresnel approximation instead leaves the pair about 2% over
    // unity on smooth dielectrics at glancing angles: two approximations of
    // the same quantity that do not cancel. Written as a complement they
    // cannot fail to.
    out.r += (1.0f - far_r) * (1.0f - mat.metallic) * ar * er;
    out.g += (1.0f - far_g) * (1.0f - mat.metallic) * ag * eg;
    out.b += (1.0f - far_b) * (1.0f - mat.metallic) * ab * eb;

    if (mat.has_brdf_lut) {
        out.r += pr * far_r;
        out.g += pg * far_g;
        out.b += pb * far_b;
    }

    // Nothing is clamped. The frame has a physical scale now -- a radiance of
    // 2 is twice a radiance of 1 -- and a ceiling here would throw that away
    // before the tone map, which is the stage that owns the range, ever saw
    // it. A sky bright enough to push a surface past white would otherwise
    // flatten every such surface to the same value and cost the object its
    // form.

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
