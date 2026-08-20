// end-to-end test of the real Engine on the CUDA path.
//
// every other test in here drives cuda_triangle.cu directly. this one drives
// Engine itself - the scene graph, node transforms, the two-pass shadow
// render, the vertex editor - because that is where the integration seams
// are, and none of the standalone tests can see them.
//
// runs headless: SDL's dummy video driver means no window and no input, so
// the loop is stepped by calling render() directly instead of run().
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include "Engine.h"

// shaders.h declares these as extern and Engine.cpp's CPU path uses them.
// main.cpp normally supplies them; keep the values identical to main.cpp,
// because the CPU shaders read these globals rather than the engine's own
// renderWidth/renderHeight.
Model *model = NULL;
Vec3f light_dir(1, 1, 1);
extern const int width = 800;
extern const int height = 800;

static int failures = 0;
static void check(bool ok, const char *what)
{
    printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

// counts non-background pixels and hashes the frame, so "did the image
// change" and "is there anything on screen" are both cheap to ask
struct FrameStat
{
    long long lit;
    unsigned long long hash;
    unsigned long long sum_r;
    unsigned long long sum_g;
    unsigned long long sum_b;
};

static FrameStat statOf(TGAImage &fb)
{
    FrameStat s = {0, 1469598103934665603ULL, 0, 0, 0};
    for (int y = 0; y < fb.get_height(); y++)
        for (int x = 0; x < fb.get_width(); x++)
        {
            TGAColor c = fb.get(x, y);
            if (c[0] || c[1] || c[2]) s.lit++;
            s.sum_b += c[0];
            s.sum_g += c[1];
            s.sum_r += c[2];
            for (int k = 0; k < 3; k++)
            {
                s.hash ^= (unsigned char)c[k];
                s.hash *= 1099511628211ULL;
            }
        }
    return s;
}

// render() leaves the frame on the GPU when it can, so pull it back the same
// way captureFrame does before inspecting it
static FrameStat renderAndStat(Engine &e, const char *save = NULL)
{
    e.render();
    const char *f = save ? save : "/tmp/_test_engine_frame.tga";
    TGAImage tmp(0, 0, TGAImage::RGB);
    e.captureFrame(f);
    tmp.read_tga_file(f);
    return statOf(tmp);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../obj/african_head.obj";

    // no window, no GPU presentation - just the software pipeline
    setenv("SDL_VIDEODRIVER", "dummy", 1);
    // the dummy video driver has no accelerated renderer, and Engine::init
    // asks for SDL_RENDERER_ACCELERATED
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, width, height);
    if (!engine.init())
    {
        printf("SKIP: engine init failed (no SDL?)\n");
        return 77;
    }
    if (!engine.isCudaAvailable())
    {
        printf("SKIP: no CUDA\n");
        return 77;
    }

    printf("\n--- scene setup\n");
    SceneNode *a = engine.loadModel(path, "A");
    if (!a || !a->hasModel())
    {
        printf("SKIP: could not load %s\n", path);
        return 77;
    }
    // a second instance of the same model, offset, so the frame exercises
    // multiple meshes and a non-identity node transform at once
    SceneNode *b = engine.loadModel(path, "B");
    if (b) b->localTransform.position = Vec3f(1.2f, 0.f, 0.f);
    check(a && b, "two mesh nodes in the scene");

    engine.toggleCudaRendering(); // starts disabled
    printf("\n--- CUDA path renders\n");

    FrameStat f1 = renderAndStat(engine, "/tmp/eng_cuda.tga");
    printf("  lit pixels: %lld\n", f1.lit);
    check(f1.lit > 2000, "CUDA path produces a non-blank frame");

    // a second identical frame must be identical: catches state that leaks
    // between frames (material slots, depth/shadow buffers not being cleared)
    FrameStat f2 = renderAndStat(engine);
    check(f1.hash == f2.hash, "two identical frames render identically");

    printf("\n--- node transforms reach the GPU\n");
    Vec3f saved = b->localTransform.position;
    b->localTransform.position = saved + Vec3f(0.6f, 0.25f, 0.f);
    FrameStat f3 = renderAndStat(engine);
    check(f3.hash != f2.hash, "moving a node changes the frame");
    b->localTransform.position = saved;
    FrameStat f4 = renderAndStat(engine);
    check(f4.hash == f2.hash, "moving it back restores the frame");

    printf("\n--- deformation invalidates the cached GPU mesh\n");
    // this is the regression: geometry was uploaded once and never refreshed,
    // so sculpting and blend shapes had no effect on the CUDA path
    Model *m = a->model;
    m->backupOriginalVertices();
    for (int i = 0; i < m->nverts(); i++)
        m->setVertex(i, m->vert(i) + Vec3f(0.f, 0.08f, 0.f));
    FrameStat f5 = renderAndStat(engine, "/tmp/eng_cuda_sculpt.tga");
    printf("  lit pixels after sculpt: %lld\n", f5.lit);
    check(f5.hash != f4.hash, "sculpting vertices changes the CUDA frame");

    m->restoreOriginalVertices();
    FrameStat f6 = renderAndStat(engine);
    check(f6.hash == f4.hash, "restoring vertices restores the frame exactly");

    // and again through the blend-shape path the expression system uses
    std::vector<Vec3f> target = m->getVertices();
    for (size_t i = 0; i < target.size(); i++)
        target[i] = target[i] + Vec3f(0.06f, 0.f, 0.f);
    m->addBlendShape("t", target);
    m->setExpressionByName("t", 1.0f);
    FrameStat f7 = renderAndStat(engine);
    check(f7.hash != f6.hash, "blend shape changes the CUDA frame");

    m->clearAllBlendWeights();
    FrameStat f8 = renderAndStat(engine);
    check(f8.hash == f6.hash, "clearing the expression restores the frame");

    printf("\n--- light controls affect both render paths\n");
    Scene &scene = engine.getScene();
    Vec3f savedLightColor = scene.light.color;
    float savedLightIntensity = scene.light.intensity;

    scene.light.intensity = 0.0f;
    FrameStat fLightOff = renderAndStat(engine);
    check(fLightOff.hash != f8.hash, "zero direct light changes the CUDA frame");
    check(fLightOff.sum_r < f8.sum_r, "zero direct light reduces CUDA brightness");

    scene.light.color = Vec3f(1.0f, 0.2f, 0.2f);
    scene.light.intensity = savedLightIntensity;
    FrameStat fWarm = renderAndStat(engine);
    check(fWarm.hash != f8.hash, "light colour changes the CUDA frame");
    check(fWarm.sum_r > fWarm.sum_b, "warm CUDA light biases the red channel");

    scene.light.color = savedLightColor;
    scene.light.intensity = savedLightIntensity;
    FrameStat fLightReset = renderAndStat(engine);
    check(fLightReset.hash == f8.hash, "restoring light settings restores the CUDA frame");

    printf("\n--- both models share one GPU mesh\n");
    // A and B are the same .obj, so Scene hands back the same Model*, and the
    // cache is keyed on that: deforming it must move both instances
    check(a->model == b->model, "both nodes share one Model (Scene caches by path)");

    printf("\n--- many meshes in one frame\n");
    // the material table used to be a fixed 64 slots, consumed by BOTH the
    // shadow pass and the colour pass, so anything past the 31st mesh was
    // silently dropped by drawMesh with no warning. 40 nodes clears that.
    for (int i = 0; i < 38; i++)
    {
        char nm[16];
        snprintf(nm, sizeof(nm), "M%d", i);
        SceneNode *n = engine.loadModel(path, nm);
        if (n)
            n->localTransform.position = Vec3f(-2.5f + 0.13f * i, -1.2f + 0.06f * i, 0.f);
    }
    FrameStat fm = renderAndStat(engine, "/tmp/eng_many.tga");
    printf("  meshes: %d, lit pixels: %lld\n", engine.getScene().getMeshCount(), fm.lit);
    check(engine.getScene().getMeshCount() == 40, "40 mesh nodes in the scene");
    check(fm.lit > f8.lit, "adding 38 more meshes adds coverage");

    // pixel coverage alone cannot prove every mesh got drawn - meshes overlap,
    // and one that lands off-screen contributes nothing either way. ask the
    // rasterizer how many triangles it was actually offered instead. under the
    // old cap drawMesh returned early, so those triangles never even reached
    // the setup kernel and the count came up short.
    int submitted = 0, cb = 0, co = 0, ovf = 0;
    cudaGetRasterStats(&submitted, &cb, &co, &ovf);
    int expect = 40 * a->model->nfaces() * 2;   // 40 meshes, shadow + colour pass
    printf("  triangles submitted: %d (expect %d)\n", submitted, expect);
    check(submitted == expect, "every mesh reaches the GPU in both passes");

    printf("\n--- CPU path still works\n");
    engine.toggleCudaRendering();
    FrameStat f9 = renderAndStat(engine, "/tmp/eng_cpu.tga");
    printf("  lit pixels: %lld\n", f9.lit);
    check(f9.lit > 2000, "CPU path still produces a non-blank frame");

    scene.light.intensity = 0.0f;
    FrameStat fCpuLightOff = renderAndStat(engine);
    check(fCpuLightOff.hash != f9.hash, "zero direct light changes the CPU frame");
    check(fCpuLightOff.sum_r < f9.sum_r, "zero direct light reduces CPU brightness");

    scene.light.color = Vec3f(1.0f, 0.2f, 0.2f);
    scene.light.intensity = savedLightIntensity;
    FrameStat fCpuWarm = renderAndStat(engine);
    check(fCpuWarm.hash != f9.hash, "light colour changes the CPU frame");
    check(fCpuWarm.sum_r > fCpuWarm.sum_b, "warm CPU light biases the red channel");

    scene.light.color = savedLightColor;
    scene.light.intensity = savedLightIntensity;

    printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
