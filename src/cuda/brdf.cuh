// The Cook-Torrance microfacet BRDF, factored out of the shader so it can be
// checked on its own.
//
// Everything here is __host__ __device__ under nvcc and plain inline under a
// host compiler, which is what lets tests/test_brdf.cpp integrate these
// functions numerically instead of taking the rendered image's word for it.
// The analytic properties they check -- that the distribution normalises, that
// the BRDF never returns more energy than it receives -- are the kind a
// differential test cannot see, because two implementations can be wrong in
// the same way and still agree.
//
// Sources: Walter et al. 2007 for GGX/Trowbridge-Reitz; Karis, "Real Shading
// in Unreal Engine 4" (SIGGRAPH 2013) for the k remap and the roughness-aware
// Fresnel; Schlick 1994 for the Fresnel approximation itself.
#pragma once

#ifdef __CUDACC__
#define BRDF_FN __host__ __device__ inline
#else
#include <math.h>
#define BRDF_FN inline
#endif

#define BRDF_PI 3.14159265358979323846f

// Roughness below this makes the GGX denominator collapse: at alpha = 0 the
// lobe is a delta function, which no point sample can hit and which divides by
// zero at n.h = 1. Every caller clamps through here rather than each picking
// its own floor.
#define BRDF_MIN_ROUGHNESS 0.045f

BRDF_FN float brdf_clamp_roughness(float rough)
{
    if (rough < BRDF_MIN_ROUGHNESS) return BRDF_MIN_ROUGHNESS;
    if (rough > 1.0f) return 1.0f;
    return rough;
}

// Trowbridge-Reitz. Roughness is perceptual and squared to alpha first, which
// is Disney's remap: it makes the visible change per slider step roughly even
// instead of piling every distinguishable appearance into the bottom tenth.
//
// Normalised so that the integral of D * cos(theta_h) over the hemisphere is
// 1. test_brdf integrates it.
BRDF_FN float brdf_D_ggx(float ndoth, float rough)
{
    if (ndoth <= 0.0f) return 0.0f;
    float a = rough * rough;
    float a2 = a * a;
    float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / (BRDF_PI * d * d);
}

BRDF_FN float brdf_G1_schlick(float ndotx, float k)
{
    float denom = ndotx * (1.0f - k) + k;
    return (denom > 1e-8f) ? ndotx / denom : 0.0f;
}

// Smith's separable masking-shadowing, with Schlick's approximation to the
// GGX term. k is Karis' remap for analytic lights; image-based lighting wants
// a different one (alpha/2), which is why prefiltering does not call this.
BRDF_FN float brdf_G_smith(float ndotv, float ndotl, float rough)
{
    float r = rough + 1.0f;
    float k = r * r * 0.125f;
    return brdf_G1_schlick(ndotv, k) * brdf_G1_schlick(ndotl, k);
}

// Smith again, with the k image-based lighting wants. The two remaps exist
// because the incoming light is different: an analytic light arrives from one
// direction, an environment from all of them at once, and Karis fits a
// separate k to each. Using the direct one for IBL over-darkens rough metals.
BRDF_FN float brdf_G_smith_ibl(float ndotv, float ndotl, float rough)
{
    float a = rough * rough;
    float k = a * 0.5f;
    return brdf_G1_schlick(ndotv, k) * brdf_G1_schlick(ndotl, k);
}

// Per channel: dielectrics have a scalar F0 near 0.04, metals take theirs from
// the albedo and so are coloured.
BRDF_FN float brdf_F_schlick(float f0, float vdoth)
{
    float m = 1.0f - vdoth;
    if (m < 0.0f) m = 0.0f;
    float m2 = m * m;
    return f0 + (1.0f - f0) * (m2 * m2 * m);
}

// Fresnel for the ambient term, where there is no single half vector to take
// v.h from and n.v stands in for it. Without the roughness ceiling a rough
// metal loses its edges: plain Schlick drives F to 1 at grazing angles, which
// on a lobe that wide is energy the surface never actually reflects.
BRDF_FN float brdf_F_schlick_roughness(float f0, float ndotv, float rough)
{
    float m = 1.0f - ndotv;
    if (m < 0.0f) m = 0.0f;
    float m2 = m * m;
    float ceiling = 1.0f - rough;
    if (ceiling < f0) ceiling = f0;
    return f0 + (ceiling - f0) * (m2 * m2 * m);
}

// The specular BRDF, D*G*F over 4*n.v*n.l, for one channel's f0. Returns the
// value of the BRDF itself; the caller supplies the incoming light and the
// cosine.
BRDF_FN float brdf_specular(float ndotv, float ndotl, float ndoth,
                            float vdoth, float rough, float f0)
{
    if (ndotv <= 0.0f || ndotl <= 0.0f) return 0.0f;
    float D = brdf_D_ggx(ndoth, rough);
    float G = brdf_G_smith(ndotv, ndotl, rough);
    float F = brdf_F_schlick(f0, vdoth);
    return (D * G * F) / (4.0f * ndotv * ndotl);
}

// ---------------------------------------------------------------------------
// The split-sum environment BRDF
// ---------------------------------------------------------------------------
// Karis' approximation splits the specular IBL integral into two averages that
// can each be precomputed: the incoming radiance, prefiltered per roughness,
// times the BRDF's response to a uniform white environment. The second factor
// depends only on n.v and roughness, and it is linear in F0 -- so it reduces to
// a scale and a bias, and one 2D table covers every material.
//
//     specular = prefiltered(R, roughness) * (F0 * A + B)
//
// brdf_env_integrate is what fills that table. It is here rather than in the
// kernel that writes the texture so tests/test_brdf.cpp can check A and B for
// energy conservation directly, without a GPU.

// Van der Corput: bit-reverse the index and read it back as a fraction. Paired
// with i/n it gives the Hammersley sequence, which spreads samples far more
// evenly than random ones at these counts.
BRDF_FN float brdf_radical_inverse(unsigned int bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return (float)bits * 2.3283064365386963e-10f;
}

// A half vector drawn from the GGX distribution, in a tangent frame where the
// normal is +Z. Sampling the distribution instead of the hemisphere is what
// makes 1024 samples enough: at low roughness almost all of the lobe's energy
// sits in a few degrees, and uniform sampling would spend its whole budget
// outside it.
BRDF_FN void brdf_importance_sample_ggx(float u1, float u2, float rough,
                                        float* hx, float* hy, float* hz)
{
    float a = rough * rough;
    float phi = 2.0f * BRDF_PI * u1;
    float cos_theta = sqrtf((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    *hx = sin_theta * cosf(phi);
    *hy = sin_theta * sinf(phi);
    *hz = cos_theta;
}

// One texel of the environment BRDF table. Returns the scale and bias to apply
// to F0. samples is the Monte Carlo budget; 1024 is where the table stops
// changing visibly.
BRDF_FN void brdf_env_integrate(float ndotv, float rough, int samples,
                                float* out_a, float* out_b)
{
    if (ndotv < 1e-4f) ndotv = 1e-4f;
    // Tangent frame with n = +Z. v is placed in the xz plane, which costs no
    // generality: the distribution is isotropic, so only the angle matters.
    float vx = sqrtf(fmaxf(0.0f, 1.0f - ndotv * ndotv));
    float vz = ndotv;

    float a = 0.0f, b = 0.0f;
    for (int i = 0; i < samples; i++)
    {
        float u1 = (float)i / (float)samples;
        float u2 = brdf_radical_inverse((unsigned int)i);

        float hx, hy, hz;
        brdf_importance_sample_ggx(u1, u2, rough, &hx, &hy, &hz);

        float vdoth = vx * hx + vz * hz;
        float lx = 2.0f * vdoth * hx - vx;
        float ly = 2.0f * vdoth * hy;
        float lz = 2.0f * vdoth * hz - vz;
        if (lz <= 0.0f) continue;

        float ndotl = lz;
        float ndoth = fmaxf(hz, 0.0f);
        float vh = fmaxf(vdoth, 0.0f);

        // The pdf and the D in the BRDF cancel, leaving only the geometry term
        // reweighted by the sampling density. This is the whole reason to
        // importance sample: D never has to be evaluated.
        float G = brdf_G_smith_ibl(ndotv, ndotl, rough);
        float g_vis = (ndoth > 1e-8f) ? (G * vh) / (ndoth * ndotv) : 0.0f;

        float m = 1.0f - vh;
        float m2 = m * m;
        float fc = m2 * m2 * m;

        a += (1.0f - fc) * g_vis;
        b += fc * g_vis;
    }

    float inv = 1.0f / (float)samples;
    *out_a = a * inv;
    *out_b = b * inv;
}

// sRGB decode. Texture maps are authored for display, so their bytes are
// already through a transfer curve; multiplying them by light without undoing
// it lights the encoding rather than the surface. Here because the material
// model is what needs it -- the tone map at the other end of the pipeline
// applies the inverse.
BRDF_FN float srgb_to_linear(float c)
{
    if (c <= 0.04045f) return c * (1.0f / 12.92f);
    return powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}
