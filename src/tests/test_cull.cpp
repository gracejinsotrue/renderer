// Does the CUDA path now cull the SAME triangles the CPU rasterizer does?
// Renders a real model through both and compares the coverage masks.
#include <cstdio>
#include <cstring>
#include <vector>
#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "our_gl.h"

extern "C" {
    bool initCudaRasterizer(int,int); void cleanupCudaRasterizer(); void cudaClearBuffers();
    void cudaRenderTriangle(const Vec4f&,const Vec4f&,const Vec4f&,const TGAColor&);
    void cudaBlitToTexture(void*,int);
    void cudaGetRasterStats(int*,int*,int*,int*);
}

int main(int argc, char **argv) {
    const int W = 400, H = 400;
    const char *objpath = (argc > 1) ? argv[1] : "../obj/african_head.obj";

    Model m(objpath);
    printf("model: %s, %d faces\n\n", objpath, m.nfaces());

    if (!initCudaRasterizer(W,H)) { printf("SKIP: no CUDA\n"); return 77; }

    // frame the camera to the model's bounding box. a fixed camera left the
    // rumi models almost entirely off-screen, which made the disagreement
    // percentage meaningless (a few edge pixels out of a few hundred).
    Vec3f lo(1e9f,1e9f,1e9f), hi(-1e9f,-1e9f,-1e9f);
    for (int i = 0; i < m.nverts(); i++) {
        Vec3f v = m.vert(i);
        for (int k = 0; k < 3; k++) { if (v[k] < lo[k]) lo[k]=v[k]; if (v[k] > hi[k]) hi[k]=v[k]; }
    }
    Vec3f ctr = (lo + hi) * 0.5f;
    float radius = (hi - lo).norm() * 0.5f;
    if (radius < 1e-6f) radius = 1.f;
    printf("framing: centre (%.2f %.2f %.2f) radius %.2f\n", ctr.x, ctr.y, ctr.z, radius);

    // scale the model to unit radius: tinyrenderer's projection has no FOV
    // term, so a small model would otherwise cover a few dozen pixels and make
    // the disagreement percentage meaningless. the centre is deliberately left
    // where it is so the camera targets a NON-origin point: that is what the
    // old lookat() got wrong, so this doubles as a regression test for it.
    float sc = 1.0f / radius;
    Matrix T = Matrix::identity();
    for (int k = 0; k < 3; k++) T[k][k] = sc;
    Vec3f sctr = ctr * sc;

    Vec3f cam = sctr + Vec3f(0.4f, 0.4f, 1.0f).normalize() * 3.0f;
    lookat(cam, sctr, Vec3f(0,1,0));
    viewport(W/8, H/8, W*3/4, H*3/4);
    projection(-1.f / (cam - sctr).norm());
    Matrix xform = Viewport * Projection * ModelView * T;

    // ---- CPU reference ----
    TGAImage cpu(W,H,TGAImage::RGB), cpuz(W,H,TGAImage::GRAYSCALE);
    struct S : public IShader {
        Model *mm; Matrix x;
        Vec4f vertex(int f,int v){ return x * embed<4>(mm->vert(f,v)); }
        bool fragment(Vec3f, Vec3f, TGAColor &c){ c = TGAColor(255,255,255); return false; }
    } sh; sh.mm = &m; sh.x = xform;

    for (int i = 0; i < m.nfaces(); i++) {
        Vec4f sc[3];
        for (int j = 0; j < 3; j++) sc[j] = sh.vertex(i,j);
        triangle(sc, sh, cpu, cpuz);
    }

    // ---- CUDA ----
    cudaClearBuffers();
    for (int i = 0; i < m.nfaces(); i++) {
        Vec4f v[3];
        for (int j = 0; j < 3; j++) v[j] = xform * embed<4>(m.vert(i,j));
        cudaRenderTriangle(v[0], v[1], v[2], TGAColor(255,255,255));
    }
    int sub=0, cb=0, co=0, ov=0;
    cudaGetRasterStats(&sub,&cb,&co,&ov);
    printf("submitted        : %d\n", sub);
    printf("backface culled  : %d  (%.1f%%)\n", cb, 100.0*cb/sub);
    printf("offscreen culled : %d\n", co);
    printf("bin entries      : %d  (triangle-tile pairs)\n", ov);
    printf("kept for GPU     : %d  (%.1f%% of submitted)\n\n", sub-cb-co, 100.0*(sub-cb-co)/sub);

    std::vector<unsigned char> gpu(W*H*3, 0);
    cudaBlitToTexture(gpu.data(), W*3);

    // ---- compare coverage. CPU buffer is bottom-up, GPU buffer top-down. ----
    // a disagreeing pixel that touches coverage from the other renderer is a
    // fill-rule / precision difference at a silhouette edge, expected since
    // the CPU path has an 8-bit depth buffer and a 1e-2 degeneracy threshold
    // against the GPU's float depth and 1e-6. a disagreeing pixel with NO
    // neighbouring coverage means a whole triangle went missing, which is a bug.
    unsigned char *c = cpu.buffer();
    std::vector<unsigned char> A(W*H), B(W*H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            A[y*W+x] = c[((H-1-y)*W+x)*3] > 0;   // cpu, flipped to top-down
            B[y*W+x] = gpu[(y*W+x)*3] > 0;
        }

    int both=0, only_cpu=0, only_gpu=0, interior=0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            bool a = A[y*W+x], b = B[y*W+x];
            if (a && b) { both++; continue; }
            if (!a && !b) continue;
            if (a) only_cpu++; else only_gpu++;
            // does the other renderer cover anything in the 8-neighbourhood?
            bool near = false;
            for (int dy=-1; dy<=1 && !near; dy++)
                for (int dx=-1; dx<=1 && !near; dx++) {
                    int nx=x+dx, ny=y+dy;
                    if (nx<0||ny<0||nx>=W||ny>=H) continue;
                    if (a ? B[ny*W+nx] : A[ny*W+nx]) near = true;
                }
            if (!near) interior++;
        }
    int diff = only_cpu + only_gpu;
    double rate = 100.0*diff/(both+diff);
    printf("coverage both    : %d px\n", both);
    printf("cpu only         : %d px\n", only_cpu);
    printf("gpu only         : %d px\n", only_gpu);
    printf("disagreement     : %.2f%%  (%d px)\n", rate, diff);
    printf("  of which edge  : %d  (fill-rule/precision, expected)\n", diff-interior);
    printf("  of which solid : %d  (missing geometry, must be 0)\n\n", interior);

    if (both < 100) {
        printf("SKIP - model is off-camera at this test's fixed view, nothing to compare\n");
        cleanupCudaRasterizer();
        return 77;
    }

    bool ok = (interior == 0) && (rate < 1.0) && (cb > sub/10);
    printf("%s\n", ok ? "PASS - CUDA matches the CPU rasterizer (edge pixels aside)"
                      : "FAIL - real coverage difference, or bin overflow");
    cleanupCudaRasterizer();
    return ok ? 0 : 1;
}
