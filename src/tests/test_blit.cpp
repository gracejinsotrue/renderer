// verifies the CUDA framebuffer lands right-way-up and in R,G,B order
#include <cstdio>
#include <cstring>
#include "geometry.h"
#include "tgaimage.h"

extern "C" {
    bool initCudaRasterizer(int width, int height);
    void cleanupCudaRasterizer();
    void cudaClearBuffers();
    void cudaRenderTriangle(const Vec4f &v0, const Vec4f &v1, const Vec4f &v2, const TGAColor &color);
    void cudaBlitToTexture(void *dst, int dst_pitch);
    void cudaCopyResults(TGAImage &framebuffer);
}

static Vec4f mk4(float x, float y, float z, float w) {
    Vec4f v; v[0]=x; v[1]=y; v[2]=z; v[3]=w; return v;
}

static void span(const unsigned char *buf, int W, int H, int stride, int &first, int &last) {
    first = last = -1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (buf[y*stride + x*3 + 0] > 0 || buf[y*stride + x*3 + 1] > 0 || buf[y*stride + x*3 + 2] > 0) {
                if (first < 0) first = y;
                last = y; break;
            }
}

int main() {
    const int W = 64, H = 64;
    if (!initCudaRasterizer(W, H)) { printf("SKIP: no CUDA device\n"); return 77; }

    cudaClearBuffers();
    // triangle sitting at LOW raster y. raster y=0 is the BOTTOM of the image
    // (TGAImage is bottom-up), so this must show up near the bottom on screen.
    // note the winding: the other order is backface-culled now
    // wound counter-clockwise in screen space, which is the front-facing
    // direction the rasterizer keeps; the other order is culled by design
    cudaRenderTriangle(mk4(2,2,10,1), mk4(40,2,10,1), mk4(2,20,10,1),
                       TGAColor(255, 0, 0));

    // --- fast path: device -> SDL-style RGB24, top-down ---
    int pitch = W*3 + 16;                       // deliberately padded, to exercise pitch
    unsigned char *sdl = new unsigned char[pitch*H];
    memset(sdl, 0, pitch*H);
    cudaBlitToTexture(sdl, pitch);

    int f, l; span(sdl, W, H, pitch, f, l);
    printf("blit : rows %d..%d (expect bottom, ~%d..%d)\n", f, l, H-1-20, H-1-2);
    bool blit_ok = (f >= H-1-21 && l <= H-1-1);

    // R must be at byte 0 for SDL_PIXELFORMAT_RGB24
    const unsigned char *px = sdl + (H-1-3)*pitch + 3*3;
    printf("blit : pixel bytes = %3d %3d %3d (expect 255 0 0)\n", px[0], px[1], px[2]);
    bool rgb_ok = (px[0] == 255 && px[1] == 0 && px[2] == 0);

    // --- fallback path: device -> TGAImage, bottom-up B,G,R ---
    TGAImage fb(W, H, TGAImage::RGB);
    cudaCopyResults(fb);
    int f2, l2; span(fb.buffer(), W, H, W*3, f2, l2);
    printf("copy : TGA rows %d..%d (expect ~2..20, i.e. raster order)\n", f2, l2);
    bool copy_ok = (f2 >= 1 && l2 <= 21);

    TGAColor c = fb.get(3, 3);
    printf("copy : TGAColor rgb = %3d %3d %3d (expect 255 0 0)\n", c[2], c[1], c[0]);
    bool copy_rgb_ok = (c[2] == 255 && c[1] == 0 && c[0] == 0);

    cleanupCudaRasterizer();
    delete[] sdl;

    bool ok = blit_ok && rgb_ok && copy_ok && copy_rgb_ok;
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
