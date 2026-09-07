// Is the HDR environment sampled along the view ray, and does the Radiance
// decoder agree with itself across both scanline encodings?
//
// The trap is that a weak assertion passes on a wrong image. "The backdrop
// changed" is satisfied by any wallpaper, and so is "the backdrop is green":
// what separates an environment from a flat image is that
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

extern "C" void cudaSetToneMapping(int);

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
static const char *UNIFORM_HDR = "t_env_uniform.hdr";
static const char *HALF_HDR = "t_env_half.hdr";
static const char *DOUBLE_HDR = "t_env_double.hdr";

static const float UNIFORM_L = 0.5f;
// bright over the +X hemisphere, near-dark over -X
static const float HALF_HI = 2.0f;
static const float HALF_LO = 0.02f;

// kind 0: constant UNIFORM_L. kind 1: split by the sign of the x component of
// the direction each texel stands for. kind 2: constant 2*UNIFORM_L, which
// exists only so the response to it can be divided by the response to kind 0.
static bool writeIBLMap(const char *path, int kind)
{
    const int W = 64, H = 32;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n", H, W);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float rgb[3];
            if (kind == 0) {
                rgb[0] = rgb[1] = rgb[2] = UNIFORM_L;
            } else if (kind == 2) {
                rgb[0] = rgb[1] = rgb[2] = 2.f * UNIFORM_L;
            } else {
                float phi = ((x + 0.5f) / W - 0.5f) * 2.f * 3.14159265f;
                float theta = (y + 0.5f) / H * 3.14159265f;
                float dx = cosf(phi) * sinf(theta);
                rgb[0] = rgb[1] = rgb[2] = (dx > 0.f) ? HALF_HI : HALF_LO;
            }
            unsigned char px[4];
            floatToRgbe(rgb, px);
            fwrite(px, 1, 4, f);
        }
    }
    fclose(f);
    return true;
}

// The colour renderScene gives an untextured mesh, which is the albedo the
// irradiance below is multiplied by. Bytes, as a display would show them:
// the shader decodes them out of sRGB before lighting them, so the linear
// albedo that reaches the arithmetic is the decode of these over 255.
static const float ALBEDO_R = 200.f, ALBEDO_G = 170.f, ALBEDO_B = 150.f;

// Written out here rather than shared with the shader on purpose: this is
// the reference the render is checked against, and a reference that calls
// the code under test cannot disagree with it.
static float srgbDecode(float c)
{
    return (c <= 0.04045f) ? c / 12.92f
                           : powf((c + 0.055f) / 1.055f, 2.4f);
}

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

    // Diffuse IBL. Everything from here on runs with the tone map off, the
    // direct light at zero and no occlusion, so what lands in the frame is the
    // irradiance term alone and can be predicted rather than eyeballed.
    printf("\n--- diffuse irradiance\n");
    if (!writeIBLMap(UNIFORM_HDR, 0) || !writeIBLMap(HALF_HDR, 1) ||
        !writeIBLMap(DOUBLE_HDR, 2)) {
        printf("could not write the IBL maps\n");
        cleanup();
        return 1;
    }
    written.push_back(UNIFORM_HDR);
    written.push_back(HALF_HDR);
    written.push_back(DOUBLE_HDR);

    cudaSetToneMapping(0);
    engine.getScene().light.intensity = 0.f;
    if (engine.isSSAOEnabled()) engine.toggleSSAO();
    engine.setExposure(1.0f);

    // The integral of cosine over a hemisphere is pi, so a constant
    // environment L convolves to exactly L and a surface renders at
    // albedo * L. That is the whole normalisation of the convolution in one
    // number: drop the sin(theta), the divide by pi, or the solid angle, and
    // this lands somewhere else.
    //
    // The prediction is a floor rather than an equality. A surface in a
    // uniform environment returns albedo * L through the diffuse lobe and a
    // little more through the specular one, which reflects what the diffuse
    // lobe did not admit: the true answer sits between albedo * L and L, and
    // approaches the floor as F0 goes to zero. For a dielectric, F0 is 0.04,
    // so the excess is a few percent of the gap. Anything outside this window
    // is not Fresnel, it is a missing factor.
    engine.loadEnvironment(UNIFORM_HDR);
    TGAImage fUni = lookFrom(engine, Vec3f(0, 0, 3), "t_env_uni.tga");
    Patch uni = centrePatch(fUni);
    report("uniform L=0.5", uni);
    float wantR = srgbDecode(ALBEDO_R / 255.f) * UNIFORM_L * 255.f;
    float wantG = srgbDecode(ALBEDO_G / 255.f) * UNIFORM_L * 255.f;
    float wantB = srgbDecode(ALBEDO_B / 255.f) * UNIFORM_L * 255.f;
    printf("  albedo * L          r=%6.1f g=%6.1f b=%6.1f (the floor)\n",
           wantR, wantG, wantB);
    // The ceiling: albedo * L plus 15% of the way to L, which is far more than
    // a 4% F0 can account for and far less than a missing factor would be.
    float ceilR = wantR + 0.15f * (UNIFORM_L * 255.f - wantR);
    float ceilG = wantG + 0.15f * (UNIFORM_L * 255.f - wantG);
    float ceilB = wantB + 0.15f * (UNIFORM_L * 255.f - wantB);
    check(uni.r >= wantR - 1.f && uni.r <= ceilR &&
          uni.g >= wantG - 1.f && uni.g <= ceilG &&
          uni.b >= wantB - 1.f && uni.b <= ceilB,
          "a constant environment convolves to itself");

    // Linearity, which needs no knowledge of the material at all: doubling
    // every texel of the environment has to double the irradiance, and every
    // factor the prediction above had to reason about divides out. This is the
    // exact half of the anchor -- the bound above pins the constant, this pins
    // the proportionality.
    engine.loadEnvironment(DOUBLE_HDR);
    TGAImage fDbl = lookFrom(engine, Vec3f(0, 0, 3), "t_env_dbl.tga");
    Patch dbl = centrePatch(fDbl);
    report("uniform L=1.0", dbl);
    float ratio = (uni.r > 1.f) ? dbl.r / uni.r : 0.f;
    printf("  ratio to L=0.5      %.4f\n", ratio);
    // 5%, because the frame is bytes: at a patch mean near 70 a single level
    // is already 1.4% of the ratio, and the RGBE encoding of the map itself
    // quantises to a 8-bit mantissa before any of this.
    check(fabsf(ratio - 2.f) < 0.05f,
          "doubling the environment doubles the irradiance");

    // With the light in one hemisphere, what the centre of the frame shows
    // depends on which way the surface there faces in WORLD space. Orbiting
    // the camera has to change it. If the eye-to-world rotation were dropped,
    // the centre normal would read as +Z in every one of these views and all
    // three would come back the same.
    engine.loadEnvironment(HALF_HDR);
    TGAImage fPX = lookFrom(engine, Vec3f(3, 0, 0), "t_env_px.tga");
    TGAImage fNX = lookFrom(engine, Vec3f(-3, 0, 0), "t_env_nx.tga");
    TGAImage fPZ = lookFrom(engine, Vec3f(0, 0, 3), "t_env_pz.tga");
    Patch px = centrePatch(fPX), nx = centrePatch(fNX), pz = centrePatch(fPZ);
    report("facing +X (lit)", px);
    report("facing -X (dark)", nx);
    report("facing +Z (half)", pz);
    // Ordering only. What the centre pixel shows is the surface the bunny
    // happens to present there, which is not axis-aligned, so the absolute
    // values are a property of the model rather than of the convolution --
    // the uniform case above is where the arithmetic is pinned down. What
    // matters here is that the three differ at all and in the right order,
    // because a dropped eye-to-world rotation makes them equal.
    check(px.r > 200.f, "a normal inside the lit hemisphere takes its radiance");
    check(px.r > 3.f * nx.r, "facing away from the light is several times darker");
    check(pz.r > nx.r + 40.f && pz.r < px.r - 40.f,
          "a normal on the boundary lands between the two");

    engine.getScene().light.intensity = 1.f;
    cudaSetToneMapping(1);
    if (!engine.isSSAOEnabled()) engine.toggleSSAO();
    engine.loadEnvironment(FLAT_HDR);

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
