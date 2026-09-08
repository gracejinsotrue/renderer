// Does supersampling anti-alias the silhouette, and does it do so without
// moving the image?
//
// The trap: "the frame changed" and "the frame got smoother" are both nearly
// free to satisfy - a blur pass would pass either one while destroying the
// render. The property that actually identifies anti-aliasing is *partial
// coverage at the silhouette*: where a 1x frame jumps from background to full
// surface in one pixel, a supersampled frame has intermediate pixels whose
// brightness is proportional to how much of the pixel the triangle covered.
//
// The second half matters just as much. Supersampling changes the viewport the
// whole pipeline renders through, so the easiest way to get it wrong is an
// image that is anti-aliased but scaled or shifted. The bounding box check
// catches that; the smoothness check alone would not.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "Engine.h"
#include "compat.h"

static int failures = 0;

static void check(bool ok, const char *what)
{
    printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        failures++;
}

static int lum(TGAImage &img, int x, int y)
{
    TGAColor c = img.get(x, y);
    return std::max((int)c[0], std::max((int)c[1], (int)c[2]));
}

// Rows where the first lit pixel is markedly dimmer than the pixel just inside
// it. That is a partially covered edge pixel, which is exactly what a
// supersampled resolve produces and a single-sample rasterizer cannot.
static int softEntries(TGAImage &img)
{
    int soft = 0;
    for (int y = 0; y < img.get_height(); y++)
        for (int x = 0; x < img.get_width() - 2; x++)
        {
            if (lum(img, x, y) == 0)
                continue;
            int a = lum(img, x, y), b = lum(img, x + 1, y);
            if (b > 20 && a < b * 4 / 5)
                soft++;
            break; // only the entry pixel of each row
        }
    return soft;
}

struct Box { int x0, y0, x1, y1, lit; };

static Box bounds(TGAImage &img)
{
    Box b = {1 << 30, 1 << 30, -1, -1, 0};
    for (int y = 0; y < img.get_height(); y++)
        for (int x = 0; x < img.get_width(); x++)
            if (lum(img, x, y) > 12)
            {
                b.x0 = std::min(b.x0, x); b.x1 = std::max(b.x1, x);
                b.y0 = std::min(b.y0, y); b.y1 = std::max(b.y1, y);
                b.lit++;
            }
    return b;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../obj/african_head.obj";

    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    // Engine::init brings up the CUDA rasterizer and returns false when there
    // is no device, so a machine without a GPU stops here, not at the
    // isCudaAvailable check below. That has to read as skipped.
    if (!engine.init()) { printf("SKIP: engine init failed (no CUDA, or no SDL)\n"); return 77; }
    if (!engine.isCudaAvailable()) { printf("SKIP: no CUDA\n"); return 77; }
    if (!engine.loadModel(path, "head")) { printf("could not load %s\n", path); return 1; }

    // SSAO would blend its own intermediate values into the edges and muddy
    // what is being measured here.
    if (engine.isSSAOEnabled()) engine.toggleSSAO();

    printf("\n--- supersampling\n");

    engine.setSSAA(1);
    check(engine.getSSAA() == 1, "SSAA reports 1x after setSSAA(1)");
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_ssaa_1x.tga");
    TGAImage a1; a1.read_tga_file("/tmp/t_ssaa_1x.tga");

    engine.setSSAA(2);
    check(engine.getSSAA() == 2, "SSAA reports 2x after setSSAA(2)");
    engine.update(); engine.render();
    engine.captureFrame("/tmp/t_ssaa_2x.tga");
    TGAImage a2; a2.read_tga_file("/tmp/t_ssaa_2x.tga");

    check(a2.get_width() == a1.get_width() && a2.get_height() == a1.get_height(),
          "the resolved frame is still the display resolution");

    Box b1 = bounds(a1), b2 = bounds(a2);
    printf("  1x bbox (%d,%d)-(%d,%d)  lit %d\n", b1.x0, b1.y0, b1.x1, b1.y1, b1.lit);
    printf("  2x bbox (%d,%d)-(%d,%d)  lit %d\n", b2.x0, b2.y0, b2.x1, b2.y1, b2.lit);

    // framing must survive the viewport change. one pixel of slack, because a
    // partially covered edge pixel can now cross the threshold either way.
    check(abs(b1.x0 - b2.x0) <= 1 && abs(b1.x1 - b2.x1) <= 1 &&
          abs(b1.y0 - b2.y0) <= 1 && abs(b1.y1 - b2.y1) <= 1,
          "supersampling does not move or rescale the image");

    // and it must not quietly lose or gain the model
    double ratio = (double)b2.lit / std::max(1, b1.lit);
    printf("  coverage ratio 2x/1x: %.3f\n", ratio);
    check(ratio > 0.95 && ratio < 1.05, "coverage is within 5%% of the 1x frame");

    int s1 = softEntries(a1), s2 = softEntries(a2);
    printf("  partially covered edge pixels: 1x %d, 2x %d\n", s1, s2);
    check(s2 > s1 * 2, "supersampling produces far more partial-coverage edges");

    // a blur would also raise that count, so pin down that the interior is not
    // being smeared: a flat patch of forehead must come back nearly unchanged.
    double d = 0;
    int n = 0;
    for (int y = 174; y <= 186; y++)
        for (int x = 394; x <= 406; x++)
        {
            d += abs(lum(a1, x, y) - lum(a2, x, y));
            n++;
        }
    d /= std::max(1, n);
    printf("  mean |diff| on a flat interior patch: %.2f\n", d);
    check(d < 6.0, "interior shading is not blurred by the resolve");

    engine.setSSAA(1);
    check(engine.getSSAA() == 1, "SSAA returns to 1x");

    engine.shutdown();
    printf("\n%s\n", failures ? "FAILED" : "all ssaa checks passed");
    return failures ? 1 : 0;
}
