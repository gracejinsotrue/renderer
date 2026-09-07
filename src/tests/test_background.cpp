// Does the background composite on the GPU, under the geometry, and does
// loading and clearing it actually reach the device?
//
// The trap this is written to avoid: checking only that the frame changed.
// A background that painted over everything, or one drawn upside down, would
// both satisfy that. The checks below pin the orientation with an asymmetric
// image and confirm the model still occludes what is behind it.
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

// Two flat bands, so an upside-down composite is not a symmetry the test can
// miss: red occupies the lower half in TGAImage row order.
static const char *BG_PATH = "/tmp/t_bg_src.tga";
static void writeBandedBackground(int w, int h)
{
    TGAImage bg(w, h, TGAImage::RGB);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            bg.set(x, y, y < h / 2 ? TGAColor(0, 0, 255) : TGAColor(0, 255, 0));
    bg.write_tga_file(BG_PATH);
}

static TGAImage frameOf(Engine &e, const char *path)
{
    e.update();
    e.render();
    e.captureFrame(path);
    TGAImage img;
    img.read_tga_file(path);
    return img;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../obj/african_head.obj";

    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    if (!engine.init()) { printf("engine init failed\n"); return 1; }
    if (!engine.isCudaAvailable()) { printf("SKIP: no CUDA\n"); return 77; }
    if (!engine.loadModel(path, "head")) { printf("could not load %s\n", path); return 1; }

    printf("\n--- no background\n");
    TGAImage plain = frameOf(engine, "/tmp/t_bg_none.tga");
    const int W = plain.get_width(), H = plain.get_height();
    check(W == 800 && H == 800, "frame is the render size");

    // a corner the head never reaches, in TGAImage row order
    TGAColor c = plain.get(20, 20);
    check(c[0] == 0 && c[1] == 0 && c[2] == 0, "corner is black with no background");

    // somewhere on the face, to confirm there is geometry to occlude with
    TGAColor faceBefore = plain.get(400, 400);
    check(faceBefore[0] || faceBefore[1] || faceBefore[2], "the model drew something");

    printf("\n--- background loaded\n");
    writeBandedBackground(256, 256);
    engine.loadBackground(BG_PATH);
    TGAImage withBg = frameOf(engine, "/tmp/t_bg_on.tga");

    // TGAColor indexes B,G,R
    TGAColor low = withBg.get(20, 20);          // lower half of the image
    TGAColor high = withBg.get(20, H - 20);     // upper half
    printf("  low  corner: %d %d %d\n", low[2], low[1], low[0]);
    printf("  high corner: %d %d %d\n", high[2], high[1], high[0]);

    check(low[0] > 200 && low[2] < 60, "lower corner takes the blue band");
    check(high[1] > 200 && high[2] < 60, "upper corner takes the green band");

    // the point of compositing under rather than over
    TGAColor faceAfter = withBg.get(400, 400);
    check(faceAfter[0] == faceBefore[0] && faceAfter[1] == faceBefore[1] &&
          faceAfter[2] == faceBefore[2],
          "geometry is unchanged where it covers the background");

    printf("\n--- background cleared\n");
    engine.getScene().clearBackground();
    TGAImage cleared = frameOf(engine, "/tmp/t_bg_off.tga");
    TGAColor back = cleared.get(20, 20);
    check(back[0] == 0 && back[1] == 0 && back[2] == 0,
          "clearing returns the corner to black");

    // a resize drops the device texture, so the engine has to re-upload it
    printf("\n--- survives a supersampling change\n");
    engine.loadBackground(BG_PATH);
    engine.setSSAA(engine.getSSAA() == 1 ? 2 : 1);
    TGAImage resized = frameOf(engine, "/tmp/t_bg_ssaa.tga");
    TGAColor r = resized.get(20, 20);
    check(r[0] > 200 && r[2] < 60, "background survives a rasterizer rebuild");

    printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
