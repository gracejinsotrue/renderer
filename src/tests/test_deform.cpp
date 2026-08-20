// regression test for stale GPU geometry.
//
// stage 2 started caching each model's geometry on the device, keyed by
// Model*, and uploaded it exactly once. nothing invalidated that cache, so
// after a sculpt or a blend shape the GPU kept drawing the original mesh.
// this covers both halves of the fix:
//   part 1 - Model::geometryVersion() changes on every write to verts_
//   part 2 - cudaUpdateMeshVerts re-uploads positions correctly, i.e. an
//            updated mesh renders identically to one built from scratch
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "our_gl.h"

extern "C" {
    bool initCudaRasterizer(int,int); void cleanupCudaRasterizer(); void cudaClearBuffers();
    void cudaCopyResults(TGAImage&);
    int  cudaCreateMesh(const float*,int,const int*,int,const float*,const float*);
    void cudaUpdateMeshVerts(int,const float*,int);
    void cudaDestroyMesh(int);
    void cudaDrawMesh(int,const float*,const float*,const float*,const float*,const float*,float,const float*,float,
                      unsigned char,unsigned char,unsigned char);
}

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

// ---------------------------------------------------------------- part 1
// every method that writes verts_ must bump the version, or the GPU copy
// silently goes stale.
static void testVersionCounter(Model& m) {
    printf("version counter\n");
    unsigned int v0 = m.geometryVersion();

    m.setVertex(0, m.vert(0) + Vec3f(0.01f, 0, 0));
    unsigned int v1 = m.geometryVersion();
    check(v1 != v0, "setVertex bumps geometryVersion");

    m.updateVertex(1, Vec3f(0.01f, 0, 0));
    unsigned int v2 = m.geometryVersion();
    check(v2 != v1, "updateVertex bumps geometryVersion");

    m.backupOriginalVertices();          // backup is not a mutation of verts_
    m.setVertex(2, m.vert(2) + Vec3f(0, 0.01f, 0));
    unsigned int v3 = m.geometryVersion();

    m.restoreOriginalVertices();
    check(m.geometryVersion() != v3, "restoreOriginalVertices bumps");

    unsigned int v4 = m.geometryVersion();
    m.resetVertices();
    check(m.geometryVersion() != v4, "resetVertices bumps");

    // blend shapes go through applyBlendShapes, the path the expression
    // system uses
    std::vector<Vec3f> target = m.getVertices();
    for (size_t i = 0; i < target.size(); i++) target[i] = target[i] + Vec3f(0, 0.05f, 0);
    m.addBlendShape("test_shape", target);
    unsigned int v5 = m.geometryVersion();
    m.setExpressionByName("test_shape", 1.0f);
    check(m.geometryVersion() != v5, "setExpressionByName bumps (via applyBlendShapes)");

    // reading must not bump - otherwise every frame re-uploads
    unsigned int v6 = m.geometryVersion();
    (void)m.vert(0); (void)m.nverts(); (void)m.getVertexData();
    check(m.geometryVersion() == v6, "reads do not bump");

    m.restoreOriginalVertices();
}

// ---------------------------------------------------------------- part 2
static const int W = 400, H = 400;

static void renderMesh(int mesh, const float* mvp, const float* clip, const float* mit,
                       const float* light, TGAImage& out) {
    cudaClearBuffers();
    float white[3]={1,1,1};
    cudaDrawMesh(mesh, mvp, clip, mit, light, white, 1.0f, NULL, 0.f, 235, 205, 185);
    cudaCopyResults(out);
}

static long long diffPixels(TGAImage& a, TGAImage& b) {
    long long d = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            TGAColor ca = a.get(x, y), cb = b.get(x, y);
            if (ca[0] != cb[0] || ca[1] != cb[1] || ca[2] != cb[2]) d++;
        }
    return d;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "../obj/african_head.obj";
    Model m(path);
    if (m.nverts() == 0) { printf("SKIP: could not load %s\n", path); return 77; }
    printf("model: %s (%d verts, %d faces)\n\n", path, m.nverts(), m.nfaces());

    testVersionCounter(m);

    if (!initCudaRasterizer(W, H)) { printf("SKIP: no CUDA\n"); return 77; }
    printf("\ngeometry re-upload\n");

    // frame the model
    Vec3f lo(1e9f,1e9f,1e9f), hi(-1e9f,-1e9f,-1e9f);
    for (int i = 0; i < m.nverts(); i++) { Vec3f v = m.vert(i);
        for (int k = 0; k < 3; k++) { if (v[k] < lo[k]) lo[k] = v[k]; if (v[k] > hi[k]) hi[k] = v[k]; } }
    Vec3f ctr = (lo + hi) * 0.5f;
    float sc = 1.f / ((hi - lo).norm() * 0.5f);
    Matrix T = Matrix::identity(); for (int k = 0; k < 3; k++) T[k][k] = sc;
    Vec3f sctr = ctr * sc;
    Vec3f cam = sctr + Vec3f(0.3f, 0.25f, 1.0f).normalize() * 3.0f;
    lookat(cam, sctr, Vec3f(0,1,0));
    viewport(W/8, H/8, W*3/4, H*3/4);
    projection(-1.f / (cam - sctr).norm());
    Matrix xf = Viewport * Projection * ModelView * T;
    Matrix clipm = Projection * ModelView * T;
    Matrix MITm = (Projection * ModelView * T).invert_transpose();
    float mvp[16], mit[16];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { mvp[r*4+c] = xf[r][c]; mit[r*4+c] = MITm[r][c]; }
    float clip[16]; for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) clip[r*4+c] = clipm[r][c];
    Vec3f lv = proj<3>(ModelView * embed<4>(Vec3f(0.4f,0.5f,1.0f).normalize(), 0.f)).normalize();
    float light[3] = { lv.x, lv.y, lv.z };

    int nf = m.nfaces();
    std::vector<int> idx(nf*3); std::vector<float> cn(nf*9), cu(nf*6);
    for (int i = 0; i < nf; i++) { std::vector<int> f = m.face(i);
        for (int j = 0; j < 3; j++) { idx[i*3+j] = f[j];
            Vec3f n = m.normal(i,j); cn[(i*3+j)*3+0]=n.x; cn[(i*3+j)*3+1]=n.y; cn[(i*3+j)*3+2]=n.z;
            Vec2f t = m.uv(i,j);     cu[(i*3+j)*2+0]=t.x; cu[(i*3+j)*2+1]=t.y; } }

    // mesh A is uploaded once and then updated in place, the way Engine does it
    int meshA = cudaCreateMesh((const float*)m.getVertexData(), m.nverts(),
                               idx.data(), nf, cn.data(), cu.data());
    TGAImage before(W,H,TGAImage::RGB);
    renderMesh(meshA, mvp, clip, mit, light, before);

    // deform: push every vertex outward from the centre. big enough to move
    // silhouette pixels, not so big it leaves the frame.
    m.backupOriginalVertices();
    for (int i = 0; i < m.nverts(); i++) {
        Vec3f v = m.vert(i);
        m.setVertex(i, v + (v - ctr) * 0.12f);
    }

    cudaUpdateMeshVerts(meshA, (const float*)m.getVertexData(), m.nverts());
    TGAImage after(W,H,TGAImage::RGB);
    renderMesh(meshA, mvp, clip, mit, light, after);

    // mesh B is built from scratch with the deformed positions - the ground
    // truth for what the updated mesh should look like
    int meshB = cudaCreateMesh((const float*)m.getVertexData(), m.nverts(),
                               idx.data(), nf, cn.data(), cu.data());
    TGAImage truth(W,H,TGAImage::RGB);
    renderMesh(meshB, mvp, clip, mit, light, truth);

    long long moved = diffPixels(before, after);
    long long wrong = diffPixels(after, truth);
    printf("  pixels changed by the deformation: %lld\n", moved);
    printf("  pixels differing from a fresh mesh: %lld\n", wrong);

    // the assertion that would have caught the original bug: with no
    // invalidation path the updated frame is byte-identical to the old one
    check(moved > 1000, "cudaUpdateMeshVerts actually changes the render");
    check(wrong == 0,   "updated mesh matches a freshly created one exactly");

    cudaDestroyMesh(meshA);
    cudaDestroyMesh(meshB);
    cleanupCudaRasterizer();

    printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
