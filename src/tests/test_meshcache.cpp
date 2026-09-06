// Does the engine's Model* -> device mesh cache survive Scene::clear()?
//
// Scene::clear() destroys every Model. The cache is keyed by Model*, so
// holding it across a clear leaks the device meshes, and the allocator is
// free to hand a newly loaded Model the address of one just freed, at which
// point the stale entry draws the old geometry.
//
// Address reuse is allocator-dependent, so asserting on pixels would give a
// test that fails only sometimes. Counting live device meshes is
// deterministic and catches the leak that is always there.
#include <cstdio>
#include <cstdlib>
#include "Engine.h"
#include "compat.h"

static int failures = 0;

static void check(bool ok, const char *what)
{
    printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        failures++;
}

static long long litPixels(const char *path)
{
    TGAImage img;
    img.read_tga_file(path);
    long long n = 0;
    for (int y = 0; y < img.get_height(); y++)
        for (int x = 0; x < img.get_width(); x++)
        {
            TGAColor c = img.get(x, y);
            if (c[0] || c[1] || c[2]) n++;
        }
    return n;
}

int main(int argc, char **argv)
{
    const char *a = (argc > 1) ? argv[1] : "../obj/african_head.obj";
    const char *b = (argc > 2) ? argv[2] : "../objs3/head.obj";

    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    if (!engine.init()) { printf("engine init failed\n"); return 1; }
    if (!engine.isCudaAvailable()) { printf("no CUDA, skipping\n"); return 0; }

    printf("\n--- one model uploaded\n");
    if (!engine.loadModel(a, "A")) { printf("could not load %s\n", a); return 1; }
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_mc_a.tga");
    int afterFirst = cudaLiveMeshCount();
    printf("  live device meshes: %d\n", afterFirst);
    check(afterFirst == 1, "one model on the device");

    printf("\n--- scene cleared, same model reloaded\n");
    engine.getScene().clear();
    if (!engine.loadModel(a, "A2")) { printf("could not reload %s\n", a); return 1; }
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_mc_a2.tga");
    int afterReload = cudaLiveMeshCount();
    printf("  live device meshes: %d\n", afterReload);
    check(afterReload == 1, "the cleared model's device mesh was released");

    // the same geometry at the same camera, so the frame has to match
    check(litPixels("/tmp/t_mc_a.tga") == litPixels("/tmp/t_mc_a2.tga"),
          "reloading the same model renders the same coverage");

    printf("\n--- scene cleared, a different model loaded\n");
    long long beforeSwap = litPixels("/tmp/t_mc_a2.tga");
    engine.getScene().clear();
    if (!engine.loadModel(b, "B")) { printf("could not load %s\n", b); return 1; }
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_mc_b.tga");
    int afterSwap = cudaLiveMeshCount();
    printf("  live device meshes: %d\n", afterSwap);
    check(afterSwap == 1, "swapping models leaves one device mesh");

    // if the stale cache entry were used, this would still be model A
    long long afterSwapLit = litPixels("/tmp/t_mc_b.tga");
    printf("  lit pixels: %lld -> %lld\n", beforeSwap, afterSwapLit);
    check(afterSwapLit > 0, "the second model drew something");
    check(afterSwapLit != beforeSwap, "a different model gives a different frame");

    printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
