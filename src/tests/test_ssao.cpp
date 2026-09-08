// Does the ambient occlusion pass darken cavities without touching flat
// surfaces, the background, or the geometry?
//
// The trap this is written to avoid: "SSAO changed the frame" is nearly free
// to satisfy and would pass on a pass that just dimmed everything uniformly.
// The checks below compare a concave region against a convex one, which is the
// property that actually distinguishes occlusion from a brightness slider.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "Engine.h"
#include "compat.h"

static int failures = 0;

static void check(bool ok, const char *what)
{
    printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        failures++;
}

// mean brightness of a small box, over pixels that actually have geometry
static double boxMean(TGAImage &img, int cx, int cy, int r)
{
    double sum = 0;
    int n = 0;
    for (int y = cy - r; y <= cy + r; y++)
        for (int x = cx - r; x <= cx + r; x++)
        {
            if (x < 0 || y < 0 || x >= img.get_width() || y >= img.get_height())
                continue;
            TGAColor c = img.get(x, y);
            int v = std::max((int)c[0], std::max((int)c[1], (int)c[2]));
            if (v > 0) { sum += v; n++; }
        }
    return n ? sum / n : 0.0;
}

static long long coverage(TGAImage &img)
{
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
    // Stays on african_head: the landmarks this test compares are coordinates
    // on that model's face -- a concave one against a convex one -- so pointing
    // it at other geometry measures nothing. The model is not tracked; fetch it
    // with tools/fetch_models.py.
    const char *path = (argc > 1) ? argv[1] : "../assets/external/african_head/african_head.obj";

    // no window, no GPU presentation. the dummy video driver has no
    // accelerated renderer and Engine::init asks for SDL_RENDERER_ACCELERATED,
    // so the software renderer has to be selected too.
    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    // Engine::init brings up the CUDA rasterizer and returns false when there
    // is no device, so a machine without a GPU stops here, not at the
    // isCudaAvailable check below. That has to read as skipped.
    if (!engine.init()) { printf("SKIP: engine init failed (no CUDA, or no SDL)\n"); return 77; }
    if (!engine.isCudaAvailable()) { printf("SKIP: no CUDA\n"); return 77; }
    if (!engine.loadModel(path, "head"))
    {
        printf("SKIP: no model at %s\n"
               "  run: python tools/fetch_models.py african_head\n", path);
        return 77;
    }

    // front view, so the landmarks below are where they are expected
    printf("\n--- ambient occlusion\n");

    if (engine.isSSAOEnabled()) engine.toggleSSAO();
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_ssao_off.tga");
    TGAImage off; off.read_tga_file("/tmp/t_ssao_off.tga");

    engine.toggleSSAO();
    check(engine.isSSAOEnabled(), "SSAO reports enabled after toggle");
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_ssao_on.tga");
    TGAImage on; on.read_tga_file("/tmp/t_ssao_on.tga");

    // Landmarks on african_head at the default camera, in TGA coordinates.
    // Re-derived after the backface cull sign was fixed: the previous pair
    // was chosen against a render that was drawing the inside of the far
    // side of the mesh, so those coordinates no longer sit on the features
    // they were named for. These two are the jaw/neck crease, which is a real
    // cavity, and the open chest, which sees the whole hemisphere.
    const int R = 6;
    double convex_off = boxMean(off, 458, 122, R);   // open chest
    double convex_on  = boxMean(on,  458, 122, R);
    double concave_off = boxMean(off, 474, 294, R);  // jaw meets neck
    double concave_on  = boxMean(on,  474, 294, R);
    printf("  chest    %.2f -> %.2f\n", convex_off, convex_on);
    printf("  jaw/neck %.2f -> %.2f\n", concave_off, concave_on);

    check(convex_off > 0 && concave_off > 0, "both landmarks have geometry on them");

    // the whole point: the cavity must lose a larger fraction than the
    // exposed surface. a pass that merely dimmed the image would fail here.
    double convex_drop = 1.0 - convex_on / std::max(1e-6, convex_off);
    double concave_drop = 1.0 - concave_on / std::max(1e-6, concave_off);
    printf("  darkening: chest %.1f%%, jaw/neck %.1f%%\n",
           convex_drop * 100.0, concave_drop * 100.0);
    check(concave_drop > convex_drop + 0.05,
          "a cavity darkens meaningfully more than an exposed surface");
    check(convex_drop < 0.25, "an exposed surface is left roughly alone");

    // occlusion is a shading term, so it must not move geometry
    check(coverage(off) == coverage(on), "SSAO does not change pixel coverage");

    // and it must leave the background completely alone
    bool bg_clean = true;
    for (int y = 0; y < 60 && bg_clean; y++)
        for (int x = 0; x < 60; x++)
        {
            TGAColor a = off.get(x, y), b = on.get(x, y);
            if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2]) { bg_clean = false; break; }
        }
    check(bg_clean, "background pixels are untouched");

    // intensity 0 has to be identical to disabled, or the toggle is lying
    engine.setSSAOIntensity(0.0f);
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_ssao_zero.tga");
    TGAImage zero; zero.read_tga_file("/tmp/t_ssao_zero.tga");
    bool same = true;
    for (int y = 0; y < zero.get_height() && same; y += 3)
        for (int x = 0; x < zero.get_width(); x += 3)
        {
            TGAColor a = off.get(x, y), b = zero.get(x, y);
            if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2]) { same = false; break; }
        }
    check(same, "intensity 0 matches SSAO disabled exactly");

    engine.shutdown();
    printf("\n%s\n", failures ? "FAILED" : "all ssao checks passed");
    return failures ? 1 : 0;
}
