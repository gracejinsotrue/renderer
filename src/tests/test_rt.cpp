// Differential test for the CUDA path tracer against the CPU one in
// ray_tracer_integration.h.
//
// A path trace cannot be compared pixel-for-pixel: the two use different RNGs,
// so only the converged average is meaningful. The geometry underneath it can
// be compared exactly though, and that is where a flattened BVH goes wrong.
// So this checks first-hit t and normal per pixel (deterministic, no RNG at
// all), then compares the shaded result statistically.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "ray_tracer_integration.h"

extern "C" {
    bool initCudaRayTracer(int, int);
    void cleanupCudaRayTracer();
    bool cudaRTSetScene(const float *, const int *, int, const float *, int);
    bool cudaRTSetSceneWithMaterials(const float *, const int *, int, const float *, int,
                                     const int *, const float *, const float *);
    void cudaRTResetAccumulation();
    void cudaRTSetCamera(const float *, const float *, const float *, const float *);
    void cudaRTSetSky(const float *, const float *);
    void cudaRTRender(int, int, int, unsigned int);
    void cudaRTGetAccum(float *);
    void cudaRTGetBVHStats(int *, int *, int *, int *);
}

static int failures = 0;
static void check(bool ok, const char *what)
{
    printf("  %-54s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

static const int W = 200, H = 200;

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../obj/african_head.obj";
    Model m(path);
    if (m.nverts() == 0) { printf("SKIP: could not load %s\n", path); return 77; }
    int nf = m.nfaces();
    printf("model: %s (%d faces)\n\n", path, nf);

    // ---- build both scenes from the same triangle soup ---------------------
    std::vector<float> verts((size_t)nf * 9);
    for (int i = 0; i < nf; i++)
        for (int j = 0; j < 3; j++)
        {
            Vec3f v = m.vert(i, j);
            verts[i * 9 + j * 3 + 0] = v.x;
            verts[i * 9 + j * 3 + 1] = v.y;
            verts[i * 9 + j * 3 + 2] = v.z;
        }

    rt_color albedo(0.7, 0.3, 0.3);
    float alb[3] = { (float)albedo.x(), (float)albedo.y(), (float)albedo.z() };

    auto material = std::make_shared<rt_lambertian>(albedo);
    rt_hittable_list world;
    for (int i = 0; i < nf; i++)
    {
        const float *p = &verts[i * 9];
        world.add(std::make_shared<rt_triangle>(
            rt_point3(p[0], p[1], p[2]),
            rt_point3(p[3], p[4], p[5]),
            rt_point3(p[6], p[7], p[8]), material));
    }
    std::shared_ptr<rt_bvh> cpu_bvh = world.build_bvh();

    if (!initCudaRayTracer(W, H)) { printf("SKIP: no CUDA\n"); return 77; }
    if (!cudaRTSetScene(verts.data(), NULL, nf, alb, 1))
    { printf("FAIL: GPU scene upload\n"); return 1; }

    int bn = 0, bl = 0, bd = 0, bt = 0;
    cudaRTGetBVHStats(&bn, &bl, &bd, &bt);
    printf("flat BVH: %d nodes, %d leaves, max depth %d, %d triangles\n\n", bn, bl, bd, bt);
    check(bn > 0 && bl > 0, "BVH built");
    check(bt == nf, "every triangle made it into the BVH");
    // a median split over n triangles with leaves of 4 should be shallow; a
    // degenerate build would show up here as a depth close to n
    check(bd < 64, "BVH depth fits the traversal stack");

    // ---- identical camera on both sides ------------------------------------
    Vec3f lo(1e9f, 1e9f, 1e9f), hi(-1e9f, -1e9f, -1e9f);
    for (int i = 0; i < m.nverts(); i++)
    {
        Vec3f v = m.vert(i);
        for (int k = 0; k < 3; k++) { if (v[k] < lo[k]) lo[k] = v[k]; if (v[k] > hi[k]) hi[k] = v[k]; }
    }
    Vec3f ctr = (lo + hi) * 0.5f;
    float radius = (hi - lo).norm() * 0.5f;

    rt_point3 lookfrom(ctr.x + radius * 0.6, ctr.y + radius * 0.4, ctr.z + radius * 2.5);
    rt_point3 lookat(ctr.x, ctr.y, ctr.z);
    rt_vec3 vup(0, 1, 0);

    double theta = degrees_to_radians(45.0);
    double h = std::tan(theta / 2);
    double viewport_height = 2 * h;
    double viewport_width = viewport_height * (double(W) / H);

    rt_vec3 wv = unit_vector(lookfrom - lookat);
    rt_vec3 uv = unit_vector(cross(vup, wv));
    rt_vec3 vv = cross(wv, uv);

    rt_vec3 viewport_u = viewport_width * uv;
    rt_vec3 viewport_v = viewport_height * -vv;
    rt_vec3 pixel_delta_u = viewport_u / W;
    rt_vec3 pixel_delta_v = viewport_v / H;
    rt_point3 upper_left = lookfrom - wv - viewport_u / 2 - viewport_v / 2;
    rt_point3 pixel00 = upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    float c_o[3]  = { (float)lookfrom.x(), (float)lookfrom.y(), (float)lookfrom.z() };
    float c_p[3]  = { (float)pixel00.x(),  (float)pixel00.y(),  (float)pixel00.z()  };
    float c_du[3] = { (float)pixel_delta_u.x(), (float)pixel_delta_u.y(), (float)pixel_delta_u.z() };
    float c_dv[3] = { (float)pixel_delta_v.x(), (float)pixel_delta_v.y(), (float)pixel_delta_v.z() };
    cudaRTSetCamera(c_o, c_p, c_du, c_dv);

    // The realtime CPU tracer returns black both on a miss and on depth
    // exhaustion, and nothing in the scene emits, so every path terminates at
    // black and the image is uniformly black. The offline rt_camera keeps a
    // sky gradient, which is the scene's only light. Use that here, otherwise
    // the shading comparison below is 0 == 0 and proves nothing.
    const rt_color SKY_LO(0.0, 0.0, 0.0), SKY_HI(0.8, 0.8, 0.8);
    float sky_lo[3] = { (float)SKY_LO.x(), (float)SKY_LO.y(), (float)SKY_LO.z() };
    float sky_hi[3] = { (float)SKY_HI.x(), (float)SKY_HI.y(), (float)SKY_HI.z() };
    cudaRTSetSky(sky_lo, sky_hi);

    // ---- 1. first-hit depth, no RNG involved -------------------------------
    printf("\nprimary-ray agreement\n");
    cudaRTResetAccumulation();
    cudaRTRender(1, 1, 2 /* depth mode */, 1u);
    std::vector<float> gpu_depth((size_t)W * H * 3);
    cudaRTGetAccum(gpu_depth.data());

    long long cpu_hits = 0, gpu_hits = 0, both = 0, only_cpu = 0, only_gpu = 0;
    double worst_t = 0.0, sum_t = 0.0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            rt_point3 target = pixel00 + (double(x) * pixel_delta_u) + (double(y) * pixel_delta_v);
            rt_ray r(lookfrom, target - lookfrom);
            rt_hit_record rec;
            bool chit = cpu_bvh->hit(r, 0.001, rt_infinity, rec);
            float gt = gpu_depth[(y * W + x) * 3];
            bool ghit = gt > 0.f;

            if (chit) cpu_hits++;
            if (ghit) gpu_hits++;
            if (chit && ghit)
            {
                both++;
                double d = std::fabs(rec.t - (double)gt);
                sum_t += d;
                if (d > worst_t) worst_t = d;
            }
            else if (chit) only_cpu++;
            else if (ghit) only_gpu++;
        }

    printf("  cpu hits %lld, gpu hits %lld, both %lld\n", cpu_hits, gpu_hits, both);
    printf("  cpu-only %lld, gpu-only %lld\n", only_cpu, only_gpu);
    printf("  mean |dt| %.3e, worst |dt| %.3e\n", both ? sum_t / both : 0.0, worst_t);

    check(cpu_hits > W * H / 20, "the model is actually in frame");
    // silhouette pixels can legitimately disagree: the CPU traverses in double
    // and the GPU in float, so a ray grazing an edge can land either way
    double disagree = 100.0 * (only_cpu + only_gpu) / (double)(W * H);
    printf("  hit/miss disagreement: %.3f%%\n", disagree);
    check(disagree < 0.5, "same first-hit coverage as the CPU BVH");
    check(worst_t < 1e-2, "same first-hit distance as the CPU BVH");

    // ---- 2. first-hit normal ------------------------------------------------
    printf("\nnormal agreement\n");
    cudaRTResetAccumulation();
    cudaRTRender(1, 1, 1 /* normal mode */, 1u);
    std::vector<float> gpu_nrm((size_t)W * H * 3);
    cudaRTGetAccum(gpu_nrm.data());

    double worst_n = 0.0, sum_n = 0.0;
    long long compared = 0, bad_n = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            rt_point3 target = pixel00 + (double(x) * pixel_delta_u) + (double(y) * pixel_delta_v);
            rt_ray r(lookfrom, target - lookfrom);
            rt_hit_record rec;
            if (!cpu_bvh->hit(r, 0.001, rt_infinity, rec)) continue;
            const float *g = &gpu_nrm[(y * W + x) * 3];
            if (g[0] == 0.f && g[1] == 0.f && g[2] == 0.f) continue;

            double dx = (g[0] * 2.0 - 1.0) - rec.normal.x();
            double dy = (g[1] * 2.0 - 1.0) - rec.normal.y();
            double dz = (g[2] * 2.0 - 1.0) - rec.normal.z();
            double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            compared++;
            sum_n += d;
            if (d > worst_n) worst_n = d;
            if (d > 0.05) bad_n++;
        }
    printf("  compared %lld, mean |dn| %.3e, worst %.3e, over-tolerance %lld\n",
           compared, compared ? sum_n / compared : 0.0, worst_n, bad_n);
    // a handful of silhouette pixels may hit different triangles on the two
    // sides, which flips the normal entirely; those are counted, not averaged
    check(compared > 0 && bad_n < compared / 200, "same first-hit normals as the CPU BVH");

    // ---- 3. converged shading ----------------------------------------------
    printf("\nshaded agreement (statistical)\n");
    const int SPP = 64, DEPTH = 4;
    cudaRTResetAccumulation();
    cudaRTRender(SPP, DEPTH, 0, 7u);
    std::vector<float> gpu_col((size_t)W * H * 3);
    cudaRTGetAccum(gpu_col.data());

    // CPU reference over the same pixels, same sample count
    std::srand(12345);
    double cpu_sum = 0.0, gpu_sum = 0.0;
    long long lit_cpu = 0, lit_gpu = 0;
    for (int y = 0; y < H; y += 2)
        for (int x = 0; x < W; x += 2)
        {
            rt_color acc(0, 0, 0);
            for (int s = 0; s < SPP; s++)
            {
                double ox = (s > 0) ? random_double() - 0.5 : 0.0;
                double oy = (s > 0) ? random_double() - 0.5 : 0.0;
                rt_point3 target = pixel00 + ((x + ox) * pixel_delta_u) + ((y + oy) * pixel_delta_v);
                rt_ray r(lookfrom, target - lookfrom);

                // same iterative form the kernel uses
                rt_color tp(1, 1, 1);
                rt_ray cur = r;
                for (int d = 0; d < DEPTH; d++)
                {
                    rt_hit_record rec;
                    if (!cpu_bvh->hit(cur, 0.001, rt_infinity, rec))
                    {
                        rt_vec3 ud = unit_vector(cur.direction());
                        double a = 0.5 * (ud.y() + 1.0);
                        tp = tp * ((1.0 - a) * SKY_LO + a * SKY_HI);
                        break;
                    }
                    rt_ray scattered;
                    rt_color att;
                    rec.mat->scatter(cur, rec.p, rec.normal, rec.front_face, att, scattered);
                    tp = tp * att;
                    cur = scattered;
                    if (d == DEPTH - 1) tp = rt_color(0, 0, 0);
                }
                acc += tp;
            }
            acc = acc / SPP;
            double cl = (acc.x() + acc.y() + acc.z()) / 3.0;
            const float *g = &gpu_col[(y * W + x) * 3];
            double gl = (g[0] + g[1] + g[2]) / 3.0;
            cpu_sum += cl; gpu_sum += gl;
            if (cl > 1e-4) lit_cpu++;
            if (gl > 1e-4) lit_gpu++;
        }

    printf("  mean radiance  cpu %.5f  gpu %.5f\n", cpu_sum / ((W / 2) * (H / 2)),
           gpu_sum / ((W / 2) * (H / 2)));
    printf("  non-black px   cpu %lld  gpu %lld\n", lit_cpu, lit_gpu);

    // guard against a vacuous pass: two black images agree perfectly and
    // prove nothing about the shading path
    check(cpu_sum > 1e-3 && lit_cpu > (W / 2) * (H / 2) / 20,
          "the CPU reference is actually producing light");

    double rel = (cpu_sum > 1e-9) ? std::fabs(cpu_sum - gpu_sum) / cpu_sum : 1.0;
    printf("  relative difference: %.2f%%\n", 100.0 * rel);
    check(rel < 0.10, "converged radiance within 10% of the CPU tracer");
    check(std::llabs(lit_cpu - lit_gpu) < (lit_cpu / 10 + 8), "similar set of lit pixels");

    // ---- 4. metal and glass -----------------------------------------------
    // Lambertian agreement above says nothing about the other two scatter
    // functions, which have their own reflect/refract maths on each side.
    struct MatCase
    {
        const char *name;
        int type;
        float fuzz;
        float ior;
    };
    const MatCase cases[2] = {
        { "metal", 1, 0.15f, 1.5f },
        { "glass", 2, 0.0f,  1.5f },
    };

    for (int ci = 0; ci < 2; ci++)
    {
        const MatCase &mc = cases[ci];
        printf("\n%s agreement (statistical)\n", mc.name);

        int types[1] = { mc.type };
        float fuzzes[1] = { mc.fuzz };
        float iors[1] = { mc.ior };
        if (!cudaRTSetSceneWithMaterials(verts.data(), NULL, nf, alb, 1,
                                         types, fuzzes, iors))
        {
            printf("  FAIL: scene upload\n");
            failures++;
            continue;
        }
        cudaRTSetCamera(c_o, c_p, c_du, c_dv);
        cudaRTSetSky(sky_lo, sky_hi);
        cudaRTResetAccumulation();
        cudaRTRender(SPP, DEPTH, 0, 11u);
        cudaRTGetAccum(gpu_col.data());

        std::shared_ptr<rt_material> cpu_mat;
        if (mc.type == 1)
            cpu_mat = std::make_shared<rt_metal>(albedo, mc.fuzz);
        else
            cpu_mat = std::make_shared<rt_dielectric>(mc.ior);

        rt_hittable_list w2;
        for (int i = 0; i < nf; i++)
        {
            const float *p = &verts[i * 9];
            w2.add(std::make_shared<rt_triangle>(
                rt_point3(p[0], p[1], p[2]), rt_point3(p[3], p[4], p[5]),
                rt_point3(p[6], p[7], p[8]), cpu_mat));
        }
        std::shared_ptr<rt_bvh> bvh2 = w2.build_bvh();

        std::srand(999);
        double csum = 0.0, gsum = 0.0;
        for (int y = 0; y < H; y += 2)
            for (int x = 0; x < W; x += 2)
            {
                rt_color acc(0, 0, 0);
                for (int s = 0; s < SPP; s++)
                {
                    double ox = (s > 0) ? random_double() - 0.5 : 0.0;
                    double oy = (s > 0) ? random_double() - 0.5 : 0.0;
                    rt_point3 target = pixel00 + ((x + ox) * pixel_delta_u) + ((y + oy) * pixel_delta_v);
                    rt_ray cur(lookfrom, target - lookfrom);
                    rt_color tp(1, 1, 1);
                    for (int d = 0; d < DEPTH; d++)
                    {
                        rt_hit_record rec;
                        if (!bvh2->hit(cur, 0.001, rt_infinity, rec))
                        {
                            rt_vec3 ud = unit_vector(cur.direction());
                            double a = 0.5 * (ud.y() + 1.0);
                            tp = tp * ((1.0 - a) * SKY_LO + a * SKY_HI);
                            break;
                        }
                        rt_ray sc;
                        rt_color att;
                        if (!rec.mat->scatter(cur, rec.p, rec.normal, rec.front_face, att, sc))
                        { tp = rt_color(0, 0, 0); break; }
                        tp = tp * att;
                        cur = sc;
                        if (d == DEPTH - 1) tp = rt_color(0, 0, 0);
                    }
                    acc += tp;
                }
                acc = acc / SPP;
                csum += (acc.x() + acc.y() + acc.z()) / 3.0;
                const float *g = &gpu_col[(y * W + x) * 3];
                gsum += (g[0] + g[1] + g[2]) / 3.0;
            }

        int px = (W / 2) * (H / 2);
        printf("  mean radiance  cpu %.5f  gpu %.5f\n", csum / px, gsum / px);
        check(csum / px > 1e-3, "the CPU reference produces light");
        double r = (csum > 1e-9) ? std::fabs(csum - gsum) / csum : 1.0;
        printf("  relative difference: %.2f%%\n", 100.0 * r);
        check(r < 0.15, "converged radiance within 15% of the CPU tracer");
    }

    cleanupCudaRayTracer();
    printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
