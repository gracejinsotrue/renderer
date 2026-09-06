// Does deferred shading actually decouple shading cost from depth complexity?
//
// This is the measurement the visibility buffer exists for, and a single
// before/after on one ordinary scene would not show it. Forward shading runs
// the fragment shader for every fragment that ever wins the depth test, so its
// cost rises with how many layers of geometry sit on a pixel. Deferred decides
// visibility first and shades once per pixel, so its cost should not move at
// all.
//
// So the test is a sweep, not a comparison: stack the same mesh N deep along
// the view axis, so every layer covers roughly the same pixels, and watch how
// each path responds as N grows. What matters is the SHAPE of the two curves.
// A single N would only show that one of them is faster today, on this scene,
// which is a much weaker claim and one that scene choice could manufacture.
//
// It also checks the two paths agree pixel for pixel at every depth. A faster
// renderer that draws something else is not a faster renderer.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cmath>

#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "transform.h"

extern "C" {
    bool initCudaRasterizer(int, int);
    void cleanupCudaRasterizer();
    void cudaClearBuffers();
    void cudaBlitToTexture(void *, int);
    int  cudaCreateMesh(const float *, int, const int *, int, const float *, const float *);
    void cudaSetMeshTexture(int, int, const unsigned char *, int, int, int);
    void cudaDrawMesh(int, const float *, const float *, const float *,
                      const float *, const float *, float, const float *, float,
                      unsigned char, unsigned char, unsigned char);
    void cudaSetDeferredShading(int);
    int  cudaGetDeferredShading();
    void cudaGetKernelTimings(float *, float *, float *);
    int  cudaGetShadeCount();
}

using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static const int W = 800, H = 800;

struct Uploaded {
    int handle;
    float mit[16];
    float light[3];
};

// Draw the mesh `layers` times, each pushed a little further along eye z, so
// the copies overlap on screen and stack in depth. Eye-space translation is
// applied between ModelView and Projection, which moves a copy toward the
// camera without moving it across the screen.
static void submit(const Uploaded &u, int layers, float step)
{
    float white[3] = {1.f, 1.f, 1.f};

    for (int i = 0; i < layers; i++) {
        Matrix T = Matrix::identity();
        T[2][3] = step * (float)i;          // eye-space z offset

        Matrix mv = T * ModelView;
        Matrix xf = Viewport * Projection * mv;
        Matrix clipm = Projection * mv;

        float mvp[16], clip[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) {
                mvp[r * 4 + c] = xf[r][c];
                clip[r * 4 + c] = clipm[r][c];
            }

        cudaDrawMesh(u.handle, mvp, clip, u.mit, u.light, white, 1.0f,
                     NULL, 0.f, 255, 255, 255);
    }
}

// One timed run at a given layer count.
//
// Whole-frame time is the wrong instrument for this question. Stacking N
// copies also multiplies the vertex transform, the binning and the coverage
// walk, and both paths pay all of that identically -- only the shading differs.
// So the frame number is mostly measuring work the change was never going to
// remove, and it buries the effect. `raster_ms` is the CUDA-event time for the
// colour stage alone: tiled_raster for forward, visibility + deferred shade
// for deferred. That is the only part where the two differ.
struct Timing {
    double frame_ms;
    double raster_ms;
    long   shades;      // fragment-shader invocations in the last frame
};

static Timing timeFrames(const Uploaded &u, int layers, float step,
                         int reps, std::vector<unsigned char> *capture)
{
    std::vector<unsigned char> px((size_t)W * H * 3);

    // warm-up, discarded: the first frame after a mode switch pays for it
    cudaClearBuffers();
    submit(u, layers, step);
    cudaBlitToTexture(px.data(), W * 3);

    double raster_total = 0.0;
    auto t0 = Clock::now();
    for (int r = 0; r < reps; r++) {
        cudaClearBuffers();
        submit(u, layers, step);
        cudaBlitToTexture(px.data(), W * 3);

        float up = 0.f, bin = 0.f, ras = 0.f;
        cudaGetKernelTimings(&up, &bin, &ras);
        raster_total += ras;
    }
    double total = ms(t0, Clock::now());

    if (capture) *capture = px;

    Timing t;
    t.frame_ms = total / reps;
    t.raster_ms = raster_total / reps;
    // counters reset in clear(), so this is one frame's worth
    t.shades = cudaGetShadeCount();
    return t;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../rumi/hair.obj";
    const int reps = 20;
    const int depths[] = {1, 2, 4, 8};
    const int ndepths = 4;

    if (!initCudaRasterizer(W, H)) { printf("SKIP: no CUDA\n"); return 77; }

    Model m(path);
    int n = m.nfaces();
    if (n <= 0) { printf("SKIP: no faces in %s\n", path); return 77; }

    lookat(Vec3f(1, 1, 3), Vec3f(0, 0, 0), Vec3f(0, 1, 0));
    viewport(W / 8, H / 8, W * 3 / 4, H * 3 / 4);
    projection(-1.f / 3.f);

    std::vector<int> idx(n * 3);
    std::vector<float> cn(n * 9), cu(n * 6);
    for (int i = 0; i < n; i++) {
        std::vector<int> f = m.face(i);
        for (int j = 0; j < 3; j++) {
            idx[i * 3 + j] = f[j];
            Vec3f nn = m.normal(i, j);
            cn[(i * 3 + j) * 3 + 0] = nn.x;
            cn[(i * 3 + j) * 3 + 1] = nn.y;
            cn[(i * 3 + j) * 3 + 2] = nn.z;
            Vec2f tt = m.uv(i, j);
            cu[(i * 3 + j) * 2 + 0] = tt.x;
            cu[(i * 3 + j) * 2 + 1] = tt.y;
        }
    }

    Uploaded u;
    u.handle = cudaCreateMesh((const float *)m.getVertexData(), m.nverts(),
                              idx.data(), n, cn.data(), cu.data());

    // Textures matter here: most of the fragment shader's cost is its texture
    // lookups, so an untextured model would understate what deferred saves.
    TGAImage *maps[3] = {&m.diffuseMap(), &m.normalMap(), &m.specularMap()};
    for (int slot = 0; slot < 3; slot++) {
        TGAImage *t = maps[slot];
        if (t->get_width() > 0 && t->get_height() > 0 && t->buffer())
            cudaSetMeshTexture(u.handle, slot, t->buffer(),
                               t->get_width(), t->get_height(),
                               t->get_bytespp());
    }

    Matrix MITm = (Projection * ModelView).invert_transpose();
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) u.mit[r * 4 + c] = MITm[r][c];

    Vec3f lv = proj<3>(ModelView * embed<4>(Vec3f(1, 1, 1).normalize())).normalize();
    u.light[0] = lv.x; u.light[1] = lv.y; u.light[2] = lv.z;

    printf("%s: %d faces, %dx%d, %d reps per point\n", path, n, W, H, reps);
    printf("stacking the mesh along eye z so the layers overlap on screen\n\n");
    printf("          shader invocations        colour stage (only part that differs)\n");
    printf("  layers    forward   deferred  ratio     forward   deferred   ratio  diff px\n");
    printf("  ------  ---------  ---------  -----   ---------  ---------  ------  -------\n");

    double f_ras0 = 0, d_ras0 = 0, f_ras1 = 0, d_ras1 = 0;

    for (int d = 0; d < ndepths; d++) {
        int layers = depths[d];
        std::vector<unsigned char> img_fwd, img_def;

        // Interleaved rather than all-forward-then-all-deferred: this machine
        // drifts enough over a run that a sequential sweep would hand the
        // trend to whichever path went second.
        cudaSetDeferredShading(0);
        Timing a = timeFrames(u, layers, 0.02f, reps, &img_fwd);
        cudaSetDeferredShading(1);
        Timing b = timeFrames(u, layers, 0.02f, reps, &img_def);
        cudaSetDeferredShading(0);
        Timing c = timeFrames(u, layers, 0.02f, reps, NULL);
        cudaSetDeferredShading(1);
        Timing e = timeFrames(u, layers, 0.02f, reps, NULL);

        double fwd_ras = 0.5 * (a.raster_ms + c.raster_ms);
        double def_ras = 0.5 * (b.raster_ms + e.raster_ms);
        double fwd_fr  = 0.5 * (a.frame_ms + c.frame_ms);
        double def_fr  = 0.5 * (b.frame_ms + e.frame_ms);

        // Same image, or the speed means nothing.
        long differing = 0;
        for (size_t i = 0; i < img_fwd.size(); i++)
            if (img_fwd[i] != img_def[i]) differing++;

        printf("  %6d  %9ld  %9ld  %4.2fx   %7.3fms  %7.3fms  %5.2fx  %7ld\n",
               layers, a.shades, b.shades,
               b.shades ? (double)a.shades / (double)b.shades : 0.0,
               fwd_ras, def_ras, fwd_ras / def_ras, differing);
        (void)fwd_fr; (void)def_fr;

        if (d == 0) { f_ras0 = fwd_ras; d_ras0 = def_ras; }
        f_ras1 = fwd_ras; d_ras1 = def_ras;
    }

    printf("\n  colour stage across 1 -> %d layers:\n", depths[ndepths - 1]);
    printf("    forward  grew %.2fx\n", f_ras1 / f_ras0);
    printf("    deferred grew %.2fx\n", d_ras1 / d_ras0);
    printf("\n  Read the invocation columns first. If forward and deferred shade\n");
    printf("  the same number of times, the scene has no overdraw and the timing\n");
    printf("  columns are answering a question that was never asked.\n");
    printf("\n  Both paths still walk every triangle to resolve coverage, so\n");
    printf("  neither timing curve can be flat. What deferred removes is the\n");
    printf("  repeated SHADING only.\n");

    cleanupCudaRasterizer();
    return 0;
}
