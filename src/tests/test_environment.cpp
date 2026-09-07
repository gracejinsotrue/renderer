// Is the HDR environment sampled along the view ray, and does the Radiance
// decoder agree with itself across both scanline encodings?
//
// The trap here is the same one test_background documents, one axis further
// out. "The backdrop changed" is satisfied by any wallpaper, and so is "the
// backdrop is green": what separates an environment from a flat image is that
// turning the camera around has to change which part of the map you are
// looking at. So the checks below move the camera to four known directions and
// demand the colour that belongs to each, rather than asserting anything about
// where colours land in the frame.
//
// The map is built here rather than shipped: an .hdr small enough to commit
// would still be a binary blob nobody can inspect, and writing it in the test
// exercises the decoder against known values.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Engine.h"
#include "hdri.h"
#include "compat.h"

static int failures = 0;

static void check(bool ok, const char *what)
{
    printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        failures++;
}

static const int ENV_W = 64;
static const int ENV_H = 32;

// Four regions with no channel in common, so a swapped axis cannot produce a
// colour that belongs to the region it should have picked.
static void envTexel(int x, int y, float *rgb)
{
    float u = (x + 0.5f) / ENV_W;
    if (y < ENV_H / 4)             { rgb[0] = 0.05f; rgb[1] = 0.05f; rgb[2] = 1.5f; }
    else if (y >= 3 * ENV_H / 4)   { rgb[0] = 1.5f;  rgb[1] = 0.05f; rgb[2] = 0.05f; }
    else if (u < 0.5f)             { rgb[0] = 0.05f; rgb[1] = 1.5f;  rgb[2] = 0.05f; }
    else                           { rgb[0] = 1.5f;  rgb[1] = 0.05f; rgb[2] = 1.5f; }
}

static void floatToRgbe(const float *rgb, unsigned char *out)
{
    float m = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2]));
    if (m < 1e-32f) { out[0] = out[1] = out[2] = out[3] = 0; return; }
    int e = 0;
    float f = frexpf(m, &e) * 256.0f / m;
    out[0] = (unsigned char)(rgb[0] * f);
    out[1] = (unsigned char)(rgb[1] * f);
    out[2] = (unsigned char)(rgb[2] * f);
    out[3] = (unsigned char)(e + 128);
}

// rle emits the (2,2,hi,lo) scanline form real files use; the flat form is the
// same pixels with no encoding at all, which is what makes the pair a check on
// the decoder rather than on itself.
static bool writeEnvHDR(const char *path, bool rle)
{
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n", ENV_H, ENV_W);

    std::vector<unsigned char> scan((size_t)ENV_W * 4);
    for (int y = 0; y < ENV_H; y++) {
        for (int x = 0; x < ENV_W; x++) {
            float rgb[3];
            envTexel(x, y, rgb);
            floatToRgbe(rgb, &scan[(size_t)x * 4]);
        }
        if (!rle) {
            fwrite(scan.data(), 1, scan.size(), f);
            continue;
        }
        unsigned char head[4] = {2, 2,
                                 (unsigned char)(ENV_W >> 8),
                                 (unsigned char)(ENV_W & 0xFF)};
        fwrite(head, 1, 4, f);
        // literal runs only. compressing would test the encoder written here
        // rather than the decoder under test.
        for (int c = 0; c < 4; c++) {
            int x = 0;
            while (x < ENV_W) {
                int n = ENV_W - x;
                if (n > 128) n = 128;
                unsigned char count = (unsigned char)n;
                fwrite(&count, 1, 1, f);
                for (int i = 0; i < n; i++)
                    fwrite(&scan[(size_t)(x + i) * 4 + c], 1, 1, f);
                x += n;
            }
        }
    }
    fclose(f);
    return true;
}

struct Patch { float r, g, b; };

// Mean of a square at the frame centre. The centre pixel is the one that
// unprojects to the view axis exactly, so it is the only place the expected
// direction is known without redoing the projection here.
static Patch centrePatch(TGAImage &img, int half = 8)
{
    int cx = img.get_width() / 2, cy = img.get_height() / 2;
    double acc[3] = {0, 0, 0};
    int n = 0;
    for (int y = cy - half; y <= cy + half; y++)
        for (int x = cx - half; x <= cx + half; x++) {
            TGAColor c = img.get(x, y);
            acc[0] += c[2];   // TGAColor indexes B,G,R
            acc[1] += c[1];
            acc[2] += c[0];
            n++;
        }
    Patch p;
    p.r = (float)(acc[0] / n);
    p.g = (float)(acc[1] / n);
    p.b = (float)(acc[2] / n);
    return p;
}

// Everything written below goes into the working directory, which is src/.
// Recorded rather than named at the end so adding a capture cannot leave one
// behind.
static std::vector<std::string> written;

static void cleanup()
{
    for (const std::string &f : written)
        remove(f.c_str());
    written.clear();
}

static TGAImage frameOf(Engine &e, const char *path)
{
    e.update();
    e.render();
    e.captureFrame(path);
    written.push_back(path);
    TGAImage img;
    img.read_tga_file(path);
    return img;
}

static TGAImage lookFrom(Engine &e, Vec3f eye, const char *path)
{
    e.getScene().camera.position = eye;
    e.getScene().camera.target = Vec3f(0, 0, 0);
    e.getScene().camera.up = Vec3f(0, 1, 0);
    return frameOf(e, path);
}

static void report(const char *what, const Patch &p)
{
    printf("  %-18s r=%6.1f g=%6.1f b=%6.1f\n", what, p.r, p.g, p.b);
}

static const char *FLAT_HDR = "t_env_flat.hdr";
static const char *RLE_HDR = "t_env_rle.hdr";

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../obj/african_head.obj";

    printf("\n--- Radiance decode\n");
    if (!writeEnvHDR(FLAT_HDR, false) || !writeEnvHDR(RLE_HDR, true)) {
        printf("could not write the test maps\n");
        return 1;
    }
    written.push_back(FLAT_HDR);
    written.push_back(RLE_HDR);

    HDRImage flat, rle;
    check(load_hdr(FLAT_HDR, flat), "flat scanlines decode");
    check(load_hdr(RLE_HDR, rle), "run-length scanlines decode");
    check(flat.width == ENV_W && flat.height == ENV_H, "dimensions come from the -Y +X line");
    check(rle.width == flat.width && rle.height == flat.height,
          "both encodings give the same size");

    if (flat.valid() && rle.valid() && flat.pixels.size() == rle.pixels.size()) {
        double worst = 0;
        for (size_t i = 0; i < flat.pixels.size(); i++)
            worst = fmax(worst, fabs((double)flat.pixels[i] - rle.pixels[i]));
        printf("  worst per-channel difference: %g\n", worst);
        check(worst == 0.0, "the two encodings decode bit-identically");
    } else {
        check(false, "the two encodings decode bit-identically");
    }

    // 1.5 is well outside 0..1, which is the whole reason for the format
    if (flat.valid()) {
        const float *top = &flat.pixels[0];
        printf("  first texel: %g %g %g\n", top[0], top[1], top[2]);
        check(top[2] > 1.2f && top[2] < 1.8f, "values above 1 survive the round trip");
        check(top[3] == 1.0f, "alpha is filled in");
    }

    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    if (!engine.init()) { printf("engine init failed\n"); cleanup(); return 1; }
    if (!engine.isCudaAvailable()) { printf("SKIP: no CUDA\n"); cleanup(); return 77; }

    // No geometry for the direction checks: the backdrop is then the entire
    // frame, so nothing can occlude the pixel being asked about.
    printf("\n--- the map is sampled along the view ray\n");
    engine.loadEnvironment(FLAT_HDR);

    TGAImage fFront = lookFrom(engine, Vec3f(0, 0, 3), "t_env_front.tga");
    TGAImage fBack = lookFrom(engine, Vec3f(0, 0, -3), "t_env_back.tga");
    TGAImage fDown = lookFrom(engine, Vec3f(0, 3, 0.6f), "t_env_down.tga");
    TGAImage fUp = lookFrom(engine, Vec3f(0, -3, 0.6f), "t_env_up.tga");
    Patch front = centrePatch(fFront);
    Patch back = centrePatch(fBack);
    Patch down = centrePatch(fDown);
    Patch up = centrePatch(fUp);
    report("looking -Z", front);
    report("looking +Z", back);
    report("looking down", down);
    report("looking up", up);

    check(front.g > 100 && front.r < 60 && front.b < 60,
          "-Z lands on the green half of the equator");
    check(back.r > 100 && back.b > 100 && back.g < 60,
          "+Z lands on the magenta half, so yaw is not ignored");
    check(down.r > 100 && down.g < 60 && down.b < 60,
          "looking down lands on the lower pole");
    check(up.b > 100 && up.r < 60 && up.g < 60,
          "looking up lands on the upper pole, so v is not flipped");

    // The map's bright channels are 1.5, which is above white. Clamping would
    // put them at exactly 255 and there would be nothing to distinguish a
    // tone curve from a clip; the curve has to leave headroom instead.
    check(front.g < 250.f && back.r < 250.f,
          "radiance above 1 rolls off rather than clipping");

    // Summed across channels rather than read off the brightest one. The
    // green here is already 241 at exposure 1, near where the curve flattens,
    // so it moves 14 levels over an eightfold exposure change while the two
    // dim channels move 135. Probing a single channel measures where on the
    // curve that channel happens to sit, not whether exposure works.
    printf("\n--- exposure\n");
    const float stops[] = {0.25f, 1.0f, 8.0f};
    float total[3];
    for (int i = 0; i < 3; i++) {
        engine.setExposure(stops[i]);
        char name[32];
        snprintf(name, sizeof(name), "t_env_e%d.tga", i);
        TGAImage f = lookFrom(engine, Vec3f(0, 0, 3), name);
        Patch p = centrePatch(f);
        total[i] = p.r + p.g + p.b;
        char label[32];
        snprintf(label, sizeof(label), "exposure %.2f", stops[i]);
        report(label, p);
    }
    check(total[1] > total[0] && total[2] > total[1],
          "brightness rises monotonically with exposure");
    check(total[2] - total[0] > 100.f, "the range covers more than rounding");
    engine.setExposure(1.0f);

    printf("\n--- under the geometry\n");
    if (!engine.loadModel(path, "subject")) {
        printf("could not load %s\n", path);
        cleanup();
        return 1;
    }
    TGAImage withModel = frameOf(engine, "t_env_model.tga");
    Patch centre = centrePatch(withModel);
    report("centre, model", centre);
    check(!(centre.g > 100 && centre.r < 60 && centre.b < 60),
          "geometry covers the backdrop where it draws");

    TGAColor corner = withModel.get(20, 20);
    check(corner[0] || corner[1] || corner[2],
          "the backdrop still draws where the model does not");

    printf("\n--- environment cleared\n");
    engine.getScene().clearEnvironment();
    TGAImage cleared = frameOf(engine, "t_env_off.tga");
    TGAColor black = cleared.get(20, 20);
    check(!black[0] && !black[1] && !black[2], "clearing returns the corner to black");

    // a resize builds a new rasterizer, which owns none of the old textures
    printf("\n--- survives a supersampling change\n");
    engine.loadEnvironment(FLAT_HDR);
    engine.setSSAA(engine.getSSAA() == 1 ? 2 : 1);
    TGAImage resized = frameOf(engine, "t_env_ssaa.tga");
    TGAColor rc = resized.get(20, 20);
    check(rc[0] || rc[1] || rc[2], "environment survives a rasterizer rebuild");

    cleanup();

    printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
