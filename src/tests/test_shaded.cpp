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
#include "our_gl.h"

extern "C" {
    bool initCudaRasterizer(int,int); void cleanupCudaRasterizer(); void cudaClearBuffers();
    void cudaCopyResults(TGAImage&);
    int  cudaCreateMesh(const float*,int,const int*,int,const float*,const float*);
    void cudaSetMeshTexture(int,int,const unsigned char*,int,int,int);
    void cudaDrawMesh(int,const float*,const float*,const float*,const float*,const float*,float,const float*,float,
                      unsigned char,unsigned char,unsigned char);
    void cudaRenderShadowPass();
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
    const char* path = (argc>1)?argv[1]:"../obj/african_head.obj";
    Model m(path);
    if(!initCudaRasterizer(W,H)){printf("SKIP: no CUDA\n");return 77;}

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
            // sign of n.l discontinuously. This reference checks the kernel
            // against ShadowMappingShader::fragment, which does not reorient,
            // so neither does this. An earlier version did, which made the
            // test agree with a kernel bug instead of catching it.
            Vec3f e = proj<3>(mit*embed<4>(nn, 0.f));
            if (e.norm()>1e-12f) e = e.normalize();
            Vec2f uv = uvc[0]*persp_bar.x + uvc[1]*persp_bar.y + uvc[2]*persp_bar.z;
            Vec3f on = mm->normal(uv);
            Vec3f en = proj<3>(mit*embed<4>(on, 0.f));
            if (en.norm()>1e-12f) { en = en.normalize(); e = en; }
            float diff = std::max(0.f, e*L);
            float rz = e.z*(2.f*(e*L)) - L.z;
            float spec = 0.f;
            if (rz>0.f) { float p = mm->specular(uv); if(p<1.f)p=1.f; spec = std::pow(rz, p); }
            TGAColor t = mm->diffuse(uv);
            float lit = 0.8f*diff + 0.3f*spec;
            c = TGAColor((unsigned char)std::min(255.f, 20.f+t[2]*lit*lc.x*li),
                         (unsigned char)std::min(255.f, 20.f+t[1]*lit*lc.y*li),
                         (unsigned char)std::min(255.f, 20.f+t[0]*lit*lc.z*li));
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

    bool ok = significant <= 64 && meanAbs <= 0.05 && maxDiff <= 96;
    printf("\n%s\n", ok ? "PASS - shaded CPU and CUDA outputs remain close"
                            : "FAIL - shaded CPU and CUDA outputs drifted too far apart");
    cleanupCudaRasterizer();
    return ok ? 0 : 1;
}