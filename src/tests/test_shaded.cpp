// differential test for the real shaded path: render the same textured model
// through the CPU reference and the CUDA rasterizer, then compare the images.
// this catches lighting/interpolation regressions that a coverage-only test is
// structurally blind to.
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "reference_raster.h"

extern "C" {
    bool initCudaRasterizer(int,int); void cleanupCudaRasterizer(); void cudaClearBuffers();
    void cudaSetToneMapping(int);
    void cudaCopyResults(TGAImage&);
    int  cudaCreateMesh(const float*,int,const int*,int,const float*,const float*);
    void cudaSetMeshTexture(int,int,const unsigned char*,int,int,int);
    void cudaDrawMesh(int,const float*,const float*,const float*,const float*,const float*,float,const float*,float,
                      unsigned char,unsigned char,unsigned char);
    void cudaRenderShadowPass();
}

// The reference BRDF, transcribed from the published equations rather than
// from src/cuda/brdf.cuh. tests/README.md is explicit about why, and this is
// the case it was written for: a reference copied out of the kernel agrees
// with the kernel by construction and cannot fail.
//
// Whether these formulas are individually right is not this test's question --
// tests/test_brdf.cpp answers that by integrating them. What this test asks is
// whether the GPU pipeline hands them the same arguments the CPU does:
// perspective-correct interpolation, the normal map through the
// inverse-transpose, the material table, and the two shading paths.
//
// Parameterised by alpha directly, where the kernel takes perceptual
// roughness and squares it internally. Same lobe, different place to put the
// square, which is the point.
static const float REF_PI = 3.14159265358979323846f;

static float refGGX(float noh, float alpha)
{
    float a2 = alpha * alpha;
    float t = noh * noh * (a2 - 1.f) + 1.f;
    return a2 / (REF_PI * t * t);
}

static float refSmithG(float nov, float nol, float k)
{
    float gv = nov / (nov * (1.f - k) + k);
    float gl = nol / (nol * (1.f - k) + k);
    return gv * gl;
}

static float refFresnel(float f0, float coh)
{
    float t = 1.f - coh;
    float t2 = t * t;
    return f0 + (1.f - f0) * t2 * t2 * t;
}

static float refFresnelRough(float f0, float nov, float rough)
{
    float t = 1.f - nov;
    float t2 = t * t;
    float ceiling = std::max(1.f - rough, f0);
    return f0 + (ceiling - f0) * t2 * t2 * t;
}

static float refSrgbDecode(float c)
{
    return (c <= 0.04045f) ? c / 12.92f
                           : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

static int diffPixels(TGAImage &a, TGAImage &b, int *maxDiff, int *significantPixels,
                      double *meanAbs)
{
    int pxdiff = 0;
    int sigdiff = 0;
    int localMax = 0;
    double absSum = 0.0;
    int samples = a.get_width() * a.get_height() * 3;
    for (int y = 0; y < a.get_height(); y++)
    {
        for (int x = 0; x < a.get_width(); x++)
        {
            TGAColor ca = a.get(x, y);
            TGAColor cb = b.get(x, y);
            bool diff = false;
            bool significant = false;
            for (int k = 0; k < 3; k++)
            {
                int d = std::abs((int)ca[k] - (int)cb[k]);
                absSum += d;
                if (d > localMax) localMax = d;
                if (d) diff = true;
                if (d > 8) significant = true;
            }
            if (diff) pxdiff++;
            if (significant) sigdiff++;
        }
    }
    if (maxDiff) *maxDiff = localMax;
    if (significantPixels) *significantPixels = sigdiff;
    if (meanAbs) *meanAbs = absSum / samples;
    return pxdiff;
}

int main(int argc,char**argv){
    const int W=600,H=600;
    // Stays on african_head: the thresholds below are calibrated against this
    // model, and the figures in the README are what they mean. It is not tracked,
    // so fetch it. Do not repoint this at a model in assets/ without
    // re-deriving the thresholds -- loosening them to fit a new model is how a
    // differential test stops testing anything.
    const char* path = (argc>1)?argv[1]:"../assets/external/african_head/african_head.obj";
    Model m(path);
    // A model that did not load renders nothing on BOTH paths, and two empty
    // images agree perfectly. Without this the test reports a confident pass
    // having compared nothing at all, which is the exact failure the suite is
    // built to catch.
    if(m.nverts()==0 || m.nfaces()==0){
        printf("SKIP: no model at %s\n  run: python tools/fetch_models.py african_head\n", path);
        return 77;
    }
    if(!initCudaRasterizer(W,H)){printf("SKIP: no CUDA\n");return 77;}
    // The reference this compares against is the shading path itself, not a
    // display of it, so the tone map is off: it wants the kernel's own numbers
    // rather than exposure and a filmic curve applied to them.
    cudaSetToneMapping(0);

    Vec3f lo(1e9f,1e9f,1e9f), hi(-1e9f,-1e9f,-1e9f);
    for(int i=0;i<m.nverts();i++){Vec3f v=m.vert(i);
        for(int k=0;k<3;k++){if(v[k]<lo[k])lo[k]=v[k];if(v[k]>hi[k])hi[k]=v[k];}}
    Vec3f ctr=(lo+hi)*0.5f; float radius=(hi-lo).norm()*0.5f;
    if (radius < 1e-6f) radius = 1.f;
    float sc=1.f/radius;
    Matrix T=Matrix::identity(); for(int k=0;k<3;k++) T[k][k]=sc;
    Vec3f sctr=ctr*sc;
    Vec3f cam=sctr+Vec3f(0.3f,0.25f,1.0f).normalize()*3.0f;

    lookat(cam,sctr,Vec3f(0,1,0));
    viewport(W/8,H/8,W*3/4,H*3/4);
    projection(-1.f/(cam-sctr).norm());
    Matrix xf=Viewport*Projection*ModelView*T;

    int nf=m.nfaces();
    std::vector<int> idx(nf*3);
    std::vector<float> cn(nf*9), cu(nf*6);
    for(int i=0;i<nf;i++){std::vector<int> f=m.face(i);
        for(int j=0;j<3;j++){ idx[i*3+j]=f[j];
            Vec3f nn=m.normal(i,j); cn[(i*3+j)*3+0]=nn.x; cn[(i*3+j)*3+1]=nn.y; cn[(i*3+j)*3+2]=nn.z;
            Vec2f tt=m.uv(i,j);     cu[(i*3+j)*2+0]=tt.x; cu[(i*3+j)*2+1]=tt.y; } }
    int mesh=cudaCreateMesh((const float*)m.getVertexData(), m.nverts(), idx.data(), nf, cn.data(), cu.data());
    if(mesh<0){printf("FAIL: mesh upload\n");return 1;}

    TGAImage* maps[3]={&m.diffuseMap(),&m.normalMap(),&m.specularMap()};
    for(int slot=0;slot<3;slot++){
        TGAImage* t=maps[slot];
        if(t->get_width()>0 && t->get_height()>0 && t->buffer())
            cudaSetMeshTexture(mesh,slot,t->buffer(),
                               t->get_width(),t->get_height(),t->get_bytespp());
    }

    Matrix MITm=(ModelView*T).invert_transpose();
    Matrix clipm=(Projection*ModelView*T);
    float mvp[16], clip[16], mit[16];
    for(int r=0;r<4;r++)for(int c=0;c<4;c++){ mvp[r*4+c]=xf[r][c]; clip[r*4+c]=clipm[r][c]; mit[r*4+c]=MITm[r][c]; }
    Vec3f lv=proj<3>(ModelView*embed<4>(Vec3f(0.4f,0.5f,1.0f).normalize(), 0.f)).normalize();
    float light[3]={lv.x,lv.y,lv.z};
    float white[3]={1,1,1};

    TGAImage gpu(W,H,TGAImage::RGB), cpu(W,H,TGAImage::RGB), cpuz(W,H,TGAImage::GRAYSCALE);

    cudaClearBuffers();
    cudaDrawMesh(mesh, mvp, clip, mit, light, white, 1.0f, NULL, 0.0f, 235,205,185);
    cudaCopyResults(gpu);

    lookat(cam,sctr,Vec3f(0,1,0));
    viewport(W/8,H/8,W*3/4,H*3/4);
    projection(-1.f/(cam-sctr).norm());
    struct S : public IShader {
        Model *mm; Matrix x; Matrix mit; Vec3f L;
        Vec3f n[3]; Vec2f uvc[3]; Vec3f lc; float li;
        Vec4f vertex(int f,int v){
            n[v]=mm->normal(f,v); uvc[v]=mm->uv(f,v);
            return x*embed<4>(mm->vert(f,v));
        }
        bool fragment(Vec3f screen_bar, Vec3f persp_bar, TGAColor &c){
            (void)screen_bar;
            Vec3f nn = n[0]*persp_bar.x + n[1]*persp_bar.y + n[2]*persp_bar.z;
            // Normals are used as they come out of MIT. Do NOT reorient them
            // toward the camera: near a silhouette an interpolated or mapped
            // normal legitimately points away, and forcing z >= 0 flips the
            // sign of n.l discontinuously. An earlier version did reorient,
            // which made the test agree with a kernel bug instead of
            // catching it.
            Vec3f e = proj<3>(mit*embed<4>(nn, 0.f));
            if (e.norm()>1e-12f) e = e.normalize();
            Vec2f uv = uvc[0]*persp_bar.x + uvc[1]*persp_bar.y + uvc[2]*persp_bar.z;
            Vec3f on = mm->normal(uv);
            Vec3f en = proj<3>(mit*embed<4>(on, 0.f));
            if (en.norm()>1e-12f) { en = en.normalize(); e = en; }
            // The view direction the kernel assumes: +Z for every fragment,
            // which is exact only under an orthographic view. The reference
            // has to make the same assumption or it is testing the projection
            // rather than the shading.
            Vec3f V(0.f, 0.f, 1.f);
            Vec3f H = (L + V).normalize();
            float nov = std::max(0.f, e.z);
            float nol = std::max(0.f, e*L);
            float noh = std::max(0.f, e*H);
            float voh = std::max(0.f, V*H);

            // The specular map holds a Phong exponent; alpha^2 = 2/(p+2) is
            // the lobe of equal width.
            float p = mm->specular(uv);
            if (p < 1.f) p = 1.f;
            float alpha = std::sqrt(2.f / (p + 2.f));
            float rough = std::sqrt(alpha);
            if (rough < 0.045f) { rough = 0.045f; alpha = rough * rough; }
            float k = (rough + 1.f) * (rough + 1.f) / 8.f;

            TGAColor t = mm->diffuse(uv);
            // Dielectric: metallic is 0 here, so F0 is the colourless 4% and
            // the whole albedo stays available to the diffuse lobe.
            const float F0 = 0.04f;
            float alb[3];
            for (int ch = 0; ch < 3; ch++) alb[ch] = refSrgbDecode(t[2-ch] / 255.f);

            float spec = 0.f;
            if (nov > 0.f && nol > 0.f)
                spec = refGGX(noh, alpha) * refSmithG(nov, nol, k)
                     * refFresnel(F0, voh) / (4.f * nov * nol);
            float kd = 1.f - refFresnel(F0, voh);

            // Irradiance of a light whose white-Lambertian response is li.
            float el = li * REF_PI;
            float ka = 1.f - refFresnelRough(F0, nov, rough);
            const float ambient = 0.03f;

            float lcv[3] = {lc.x, lc.y, lc.z};
            unsigned char outc[3];
            for (int ch = 0; ch < 3; ch++) {
                float v = (kd * alb[ch] / REF_PI + spec) * nol * el * lcv[ch]
                        + ka * alb[ch] * ambient;
                outc[ch] = (unsigned char)std::min(255.f, std::max(0.f, v) * 255.f);
            }
            c = TGAColor(outc[0], outc[1], outc[2]);
            return false;
        }
    } sh;
    sh.mm=&m; sh.x=xf; sh.mit=MITm; sh.L=lv; sh.lc=Vec3f(1,1,1); sh.li=1.0f;
    for(int i=0;i<nf;i++){ Vec4f scv[3]; for(int j=0;j<3;j++) scv[j]=sh.vertex(i,j); triangle(scv, sh, cpu, cpuz); }

    int maxDiff = 0;
    int significant = 0;
    double meanAbs = 0.0;
    int pxdiff = diffPixels(cpu, gpu, &maxDiff, &significant, &meanAbs);
    double rate = 100.0 * pxdiff / (W * H);
    double srate = 100.0 * significant / (W * H);
    printf("pixels_different=%d (%.3f%%)\n", pxdiff, rate);
    printf("significant_pixels=%d (%.3f%%, >8 in any channel)\n", significant, srate);
    printf("mean_abs_byte_diff=%.4f\n", meanAbs);
    printf("max_byte_diff=%d\n", maxDiff);

    // meanAbs counts every pixel that differs at all, including by one LSB,
    // so it tracks how much of the model is textured more than how far the
    // two rasterizers disagree. `significant` is the real check: pixels off by
    // more than 8 in any channel, of which there is 1.
    //
    // These are tighter than the numbers the Phong version carried (64 / 0.08
    // / 96 against an observed 2 / 0.06 / 20). Not because anything was
    // loosened or tuned: the microfacet BRDF has no pow(rz, p) in it, and that
    // exponent was where the two paths used to diverge -- at p = 255 a
    // last-place difference in rz becomes a visible one in the result. The
    // headroom is deliberate all the same. Do not close it to fit a
    // measurement; a threshold set to whatever the last run produced fails on
    // the next driver.
    bool ok = significant <= 16 && meanAbs <= 0.06 && maxDiff <= 48;
    printf("\n%s\n", ok ? "PASS - shaded CPU and CUDA outputs remain close"
                            : "FAIL - shaded CPU and CUDA outputs drifted too far apart");
    cleanupCudaRasterizer();
    return ok ? 0 : 1;
}