// the mesh path should reject triangles that are wholly behind the camera
// before screen-space bbox/binning work. this is the failure mode that the old
// offscreen bbox check could miss: negative w can still produce nonsense x/w.
#include <cstdio>
#include <vector>
#include <algorithm>
#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "transform.h"

extern "C" {
    bool initCudaRasterizer(int,int); void cleanupCudaRasterizer(); void cudaClearBuffers();
    void cudaSetToneMapping(int);
    void cudaBlitToTexture(void*,int); void cudaGetRasterStats(int*,int*,int*,int*);
    int  cudaCreateMesh(const float*,int,const int*,int,const float*,const float*);
    void cudaDrawMesh(int,const float*,const float*,const float*,const float*,const float*,float,const float*,float,
                      unsigned char,unsigned char,unsigned char);
}

static int litPixels(const std::vector<unsigned char>& fb) {
    int lit = 0;
    for (size_t i = 0; i + 2 < fb.size(); i += 3)
        if (fb[i] || fb[i + 1] || fb[i + 2]) lit++;
    return lit;
}

int main(int argc, char** argv) {
    const int W = 400, H = 400;
    const char* path = (argc > 1) ? argv[1] : "../obj/african_head.obj";
    Model m(path);
    if (!initCudaRasterizer(W, H)) { printf("SKIP: no CUDA\n"); return 77; }
    // The reference this compares against is the shading path itself, not a
    // display of it, so the tone map is off: it wants the kernel's own numbers
    // rather than exposure and a filmic curve applied to them.
    cudaSetToneMapping(0);

    Vec3f lo(1e9f,1e9f,1e9f), hi(-1e9f,-1e9f,-1e9f);
    for (int i = 0; i < m.nverts(); i++) {
        Vec3f v = m.vert(i);
        for (int k = 0; k < 3; k++) { if (v[k] < lo[k]) lo[k] = v[k]; if (v[k] > hi[k]) hi[k] = v[k]; }
    }
    Vec3f ctr = (lo + hi) * 0.5f;
    float radius = (hi - lo).norm() * 0.5f;
    if (radius < 1e-6f) radius = 1.f;
    float sc = 1.f / radius;
    Matrix S = Matrix::identity(); for (int k = 0; k < 3; k++) S[k][k] = sc;
    Vec3f sctr = ctr * sc;
    Vec3f cam = sctr + Vec3f(0.4f, 0.4f, 1.0f).normalize() * 3.0f;
    lookat(cam, sctr, Vec3f(0,1,0));
    viewport(W/8, H/8, W*3/4, H*3/4);
    projection(-1.f / (cam - sctr).norm());

    int nf = m.nfaces();
    std::vector<int> idx(nf * 3);
    std::vector<float> cn(nf * 9), cu(nf * 6);
    for (int i = 0; i < nf; i++) { std::vector<int> f = m.face(i);
        for (int j = 0; j < 3; j++) { idx[i * 3 + j] = f[j];
            Vec3f nn = m.normal(i, j); cn[(i * 3 + j) * 3 + 0] = nn.x; cn[(i * 3 + j) * 3 + 1] = nn.y; cn[(i * 3 + j) * 3 + 2] = nn.z;
            Vec2f tt = m.uv(i, j);     cu[(i * 3 + j) * 2 + 0] = tt.x; cu[(i * 3 + j) * 2 + 1] = tt.y; } }
    int mesh = cudaCreateMesh((const float*)m.getVertexData(), m.nverts(), idx.data(), nf, cn.data(), cu.data());
    if (mesh < 0) { printf("FAIL: mesh upload\n"); return 1; }

    Matrix onClip = Projection * ModelView * S;
    Matrix onMvp = Viewport * onClip;
    Matrix MITm = onClip.invert_transpose();
    float on_clip[16], on_mvp[16], mit[16];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
        on_clip[r * 4 + c] = onClip[r][c];
        on_mvp[r * 4 + c] = onMvp[r][c];
        mit[r * 4 + c] = MITm[r][c];
    }
    Vec3f lv = proj<3>(ModelView * embed<4>(Vec3f(1,1,1).normalize(), 0.f)).normalize();
    float light[3] = {lv.x, lv.y, lv.z};
    float white[3] = {1,1,1};
    std::vector<unsigned char> fb(W * H * 3, 0);

    cudaClearBuffers();
    cudaDrawMesh(mesh, on_mvp, on_clip, mit, light, white, 1.0f, NULL, 0.f, 255,255,255);
    cudaBlitToTexture(fb.data(), W * 3);
    int sub = 0, cb = 0, co = 0, bins = 0;
    cudaGetRasterStats(&sub, &cb, &co, &bins);
    int litOn = litPixels(fb);
    printf("on-screen  : lit=%d submitted=%d culled_back=%d culled_off=%d\n", litOn, sub, cb, co);

    Matrix T = Matrix::identity();
    Vec3f push = (cam - sctr) * 2.5f;
    T[0][3] = push.x; T[1][3] = push.y; T[2][3] = push.z;
    Matrix offClip = Projection * ModelView * T * S;
    Matrix offMvp = Viewport * offClip;
    float off_clip[16], off_mvp[16];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
        off_clip[r * 4 + c] = offClip[r][c];
        off_mvp[r * 4 + c] = offMvp[r][c];
    }

    std::fill(fb.begin(), fb.end(), 0);
    cudaClearBuffers();
    cudaDrawMesh(mesh, off_mvp, off_clip, mit, light, white, 1.0f, NULL, 0.f, 255,255,255);
    cudaBlitToTexture(fb.data(), W * 3);
    int sub2 = 0, cb2 = 0, co2 = 0, bins2 = 0;
    cudaGetRasterStats(&sub2, &cb2, &co2, &bins2);
    int litOff = litPixels(fb);
    printf("behind cam : lit=%d submitted=%d culled_back=%d culled_off=%d\n", litOff, sub2, cb2, co2);

    bool ok = litOn > 1000 && litOff == 0 && co2 > co;
    printf("\n%s\n", ok ? "PASS - behind-camera mesh is rejected before raster work"
                            : "FAIL - behind-camera mesh survived frustum culling");
    cleanupCudaRasterizer();
    return ok ? 0 : 1;
}