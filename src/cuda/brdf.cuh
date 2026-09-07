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
