// Analytic checks on the microfacet BRDF, by numerical integration.
//
// This is the one test in the suite that needs neither a GPU nor a model, and
// it is deliberately not a differential test. A differential test compares two
// implementations, so it is blind to any error both of them make -- and a
// missing normalisation factor or a Fresnel term applied twice is exactly the
// kind of error that survives being written out a second time by the same
// person. What it cannot survive is integration: a distribution that does not
// integrate to one is wrong no matter how many implementations agree on it.
//
// So the properties below come from the definitions rather than from any
// rendering: the GGX distribution is a probability density over half vectors
// and must normalise; a passive surface cannot reflect more light than reaches
// it; Fresnel must equal F0 head-on and reach 1 at grazing.
#include <cstdio>
#include <cmath>

#include "cuda/brdf.cuh"

static int failures = 0;

static void check(bool cond, const char *what)
{
    printf("  %s %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond) failures++;
}

static void checkNear(double got, double want, double tol, const char *what)
{
    bool ok = std::fabs(got - want) <= tol;
    printf("  %s %s (got %.6f, want %.6f +- %.6f)\n",
           ok ? "ok  " : "FAIL", what, got, want, tol);
    if (!ok) failures++;
}

// The integral of D(h) * (n.h) over the hemisphere, which is 1 for a
// normalised microfacet distribution: D is a density of facet area per unit
// solid angle, and projecting it back onto the surface has to recover the
// surface. Only the polar angle matters, so this is a 1D quadrature times the
// 2*pi of azimuth.
static double integrateD(float rough, int steps)
{
    double sum = 0.0;
    double dtheta = (BRDF_PI * 0.5) / steps;
    for (int i = 0; i < steps; i++)
    {
        double theta = (i + 0.5) * dtheta;
        double ct = std::cos(theta), st = std::sin(theta);
        sum += brdf_D_ggx((float)ct, rough) * ct * st * dtheta;
    }
    return sum * 2.0 * BRDF_PI;
}

// Directional albedo: of the light arriving from direction v, what fraction
// does the specular lobe send back out across the whole hemisphere? A passive
// surface cannot exceed 1. Computed with f0 = 1 so Fresnel takes nothing out
// and the geometry term is the only thing that can lose energy -- which is the
// worst case, and the one worth bounding.
static double specularAlbedo(float rough, float theta_v, int steps)
{
    double vx = std::sin(theta_v), vz = std::cos(theta_v);
    double ndotv = vz;
    double sum = 0.0;
    double dtheta = (BRDF_PI * 0.5) / steps;
    double dphi = (2.0 * BRDF_PI) / (2 * steps);

    for (int i = 0; i < steps; i++)
    {
        double theta = (i + 0.5) * dtheta;
        double ct = std::cos(theta), st = std::sin(theta);
        for (int j = 0; j < 2 * steps; j++)
        {
            double phi = (j + 0.5) * dphi;
            double lx = st * std::cos(phi), ly = st * std::sin(phi), lz = ct;

            double hx = lx + vx, hy = ly, hz = lz + vz;
            double hl = std::sqrt(hx * hx + hy * hy + hz * hz);
            if (hl < 1e-12) continue;
            hx /= hl; hy /= hl; hz /= hl;

            double vdoth = vx * hx + vz * hz;
            double f = brdf_specular((float)ndotv, (float)ct, (float)hz,
                                     (float)vdoth, rough, 1.0f);
            sum += f * ct * st * dtheta * dphi;
        }
    }
    return sum;
}

int main()
{
    printf("--- GGX normalisation: integral of D * cos over the hemisphere\n");
    // 4096 steps of midpoint quadrature. The tolerance is loosest at low
    // roughness, where the lobe is narrow enough that the quadrature itself is
    // the limiting error rather than the distribution.
    const float roughs[] = {0.1f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (int i = 0; i < 5; i++)
    {
        char label[64];
        snprintf(label, sizeof(label), "roughness %.2f normalises", roughs[i]);
        checkNear(integrateD(roughs[i], 4096), 1.0,
                  roughs[i] < 0.2f ? 5e-3 : 1e-3, label);
    }

    printf("\n--- energy: specular albedo stays under 1 with f0 = 1\n");
    for (int i = 0; i < 5; i++)
    {
        for (int a = 0; a < 3; a++)
        {
            float theta_v = (float)(a * 0.35 + 0.05);
            double alb = specularAlbedo(roughs[i], theta_v, 128);
            char label[96];
            snprintf(label, sizeof(label),
                     "roughness %.2f at %.0f deg reflects %.4f of what arrives",
                     roughs[i], theta_v * 180.0 / BRDF_PI, alb);
            check(alb > 0.0 && alb <= 1.0, label);
        }
    }

    printf("\n--- Fresnel\n");
    checkNear(brdf_F_schlick(0.04f, 1.0f), 0.04, 1e-6,
              "head-on reflectance is F0");
    checkNear(brdf_F_schlick(0.04f, 0.0f), 1.0, 1e-6,
              "grazing reflectance is 1");
    checkNear(brdf_F_schlick(1.0f, 0.5f), 1.0, 1e-6,
              "F0 = 1 reflects everything at every angle");
    {
        bool monotone = true;
        float prev = brdf_F_schlick(0.04f, 1.0f);
        for (int i = 19; i >= 0; i--)
        {
            float f = brdf_F_schlick(0.04f, i / 20.0f);
            if (f < prev - 1e-6f) monotone = false;
            prev = f;
        }
        check(monotone, "rises monotonically from head-on to grazing");
    }
    // The roughness-aware form is what the ambient term uses, and its whole
    // purpose is to stop short of 1 as roughness climbs.
    check(brdf_F_schlick_roughness(0.04f, 0.0f, 1.0f) < 0.05f,
          "roughness 1 does not drive ambient Fresnel to 1 at grazing");
    checkNear(brdf_F_schlick_roughness(0.04f, 1.0f, 0.5f), 0.04, 1e-6,
              "roughness-aware form still equals F0 head-on");

    printf("\n--- masking-shadowing\n");
    {
        bool bounded = true, shrinks = true;
        float prev_at_grazing = 1.0f;
        for (int i = 0; i <= 10; i++)
        {
            float r = i / 10.0f;
            for (int j = 1; j <= 10; j++)
            {
                float c = j / 10.0f;
                float g = brdf_G_smith(c, c, r);
                if (g < 0.0f || g > 1.0f) bounded = false;
            }
            float g_graze = brdf_G_smith(0.1f, 0.1f, r);
            if (g_graze > prev_at_grazing + 1e-6f) shrinks = false;
            prev_at_grazing = g_graze;
        }
        check(bounded, "G stays within 0..1");
        check(shrinks, "G falls as roughness rises, at a fixed grazing angle");
        checkNear(brdf_G_smith(1.0f, 1.0f, 0.0f), 1.0, 1e-6,
                  "nothing is masked head-on on a smooth surface");
    }

    printf("\n--- roughness clamp\n");
    checkNear(brdf_clamp_roughness(0.0f), BRDF_MIN_ROUGHNESS, 1e-9,
              "zero clamps to the floor");
    checkNear(brdf_clamp_roughness(2.0f), 1.0, 1e-9, "above 1 clamps to 1");
    // The floor exists because alpha = 0 makes the GGX denominator vanish at
    // n.h = 1. Confirm the clamped value is finite there, which is the actual
    // failure it prevents.
    {
        float d = brdf_D_ggx(1.0f, brdf_clamp_roughness(0.0f));
        check(std::isfinite(d) && d > 0.0f,
              "the clamped floor keeps D finite at n.h = 1");
    }

    printf("\n--- sRGB decode\n");
    checkNear(srgb_to_linear(0.0f), 0.0, 1e-9, "black stays black");
    checkNear(srgb_to_linear(1.0f), 1.0, 1e-6, "white stays white");
    // The value everyone quotes: middle grey on a display is about a fifth of
    // the light of white, not half of it. This is the whole reason the decode
    // is here, so it is worth pinning to a number.
    checkNear(srgb_to_linear(0.5f), 0.2140, 1e-3, "0.5 decodes to about 0.214");
    {
        bool monotone = true;
        float prev = -1.f;
        for (int i = 0; i <= 100; i++)
        {
            float v = srgb_to_linear(i / 100.0f);
            if (v < prev) monotone = false;
            prev = v;
        }
        check(monotone, "decode is monotonic");
    }

    printf("\n%s\n", failures == 0 ? "PASS - BRDF satisfies its analytic properties"
                                   : "FAIL - BRDF violates an analytic property");
    return failures == 0 ? 0 : 1;
}
