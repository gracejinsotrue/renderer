// stage 2 validation: geometry resident on the device and transformed by
// mesh_setup_kernel must produce the same picture as the old host path that
// transformed every face on the CPU and re-uploaded it each frame.
#include <cstdio>
#include <vector>
#include "geometry.h"
#include "tgaimage.h"
#include "model.h"
#include "transform.h"
extern "C" {
    bool initCudaRasterizer(int,int); void cleanupCudaRasterizer(); void cudaClearBuffers();
    void cudaRenderTriangle(const Vec4f&,const Vec4f&,const Vec4f&,const TGAColor&);
    void cudaBlitToTexture(void*,int); void cudaGetRasterStats(int*,int*,int*,int*);
    int  cudaCreateMesh(const float*,int,const int*,int,const float*,const float*);
    void cudaSetMeshTexture(int,int,const unsigned char*,int,int,int);
    void cudaDrawMesh(int,const float*,const float*,const float*,const float*,const float*,float,const float*,float,unsigned char,unsigned char,unsigned char);
    void cudaRenderShadowPass();
}

int main(int argc,char**argv){
    const int W=400,H=400;
    const char* path = (argc>1)?argv[1]:"../obj/african_head.obj";
    Model m(path);
    if(!initCudaRasterizer(W,H)){printf("SKIP: no CUDA\n");return 77;}

    Vec3f lo(1e9f,1e9f,1e9f), hi(-1e9f,-1e9f,-1e9f);
    for(int i=0;i<m.nverts();i++){Vec3f v=m.vert(i);
        for(int k=0;k<3;k++){if(v[k]<lo[k])lo[k]=v[k];if(v[k]>hi[k])hi[k]=v[k];}}
    Vec3f ctr=(lo+hi)*0.5f; float radius=(hi-lo).norm()*0.5f;
    float sc=1.f/radius;
    Matrix T=Matrix::identity(); for(int k=0;k<3;k++) T[k][k]=sc;
    Vec3f sctr=ctr*sc;
    Vec3f cam=sctr+Vec3f(0.4f,0.4f,1.0f).normalize()*3.0f;
    lookat(cam,sctr,Vec3f(0,1,0));
    viewport(W/8,H/8,W*3/4,H*3/4);
    projection(-1.f/(cam-sctr).norm());
    Matrix xf=Viewport*Projection*ModelView*T;

    // the mesh is uploaded in model space; T folds into the matrix
    int nf=m.nfaces();
    std::vector<int> idx(nf*3);
    std::vector<float> cn(nf*9), cu(nf*6);
    for(int i=0;i<nf;i++){std::vector<int> f=m.face(i);
        for(int j=0;j<3;j++){ idx[i*3+j]=f[j];
            Vec3f nn=m.normal(i,j); cn[(i*3+j)*3+0]=nn.x; cn[(i*3+j)*3+1]=nn.y; cn[(i*3+j)*3+2]=nn.z;
            Vec2f tt=m.uv(i,j);     cu[(i*3+j)*2+0]=tt.x; cu[(i*3+j)*2+1]=tt.y; } }
    int mesh=cudaCreateMesh((const float*)m.getVertexData(), m.nverts(), idx.data(), nf, cn.data(), cu.data());
    if(mesh<0){printf("FAIL: mesh upload\n");return 1;}
    float mvp[16]; for(int r=0;r<4;r++)for(int c=0;c<4;c++) mvp[r*4+c]=xf[r][c];
    Matrix clipm=(Projection*ModelView*T);
    float clip[16]; for(int r=0;r<4;r++)for(int c=0;c<4;c++) clip[r*4+c]=clipm[r][c];
    Matrix MITm=(Projection*ModelView*T).invert_transpose();
    float mit[16]; for(int r=0;r<4;r++)for(int c=0;c<4;c++) mit[r*4+c]=MITm[r][c];
    Vec3f lv=proj<3>(ModelView*embed<4>(Vec3f(1,1,1).normalize())).normalize();
    float light[3]={lv.x,lv.y,lv.z};

    std::vector<unsigned char> A(W*H*3,0), B(W*H*3,0);

    // --- A: GPU-resident mesh, one draw call ---
    cudaClearBuffers();
    float white[3]={1,1,1};
    cudaDrawMesh(mesh, mvp, clip, mit, light, white, 1.0f, NULL, 0.f, 255,255,255);
    cudaBlitToTexture(A.data(), W*3);
    int sa=0,ca=0,oa=0,ea=0; cudaGetRasterStats(&sa,&ca,&oa,&ea);

    // --- B: host transforms every face, uploads per frame (the old path) ---
    cudaClearBuffers();
    for(int i=0;i<nf;i++){
        Vec4f v[3]; for(int j=0;j<3;j++) v[j]=xf*embed<4>(m.vert(i,j));
        cudaRenderTriangle(v[0],v[1],v[2],TGAColor(255,255,255));
    }
    cudaBlitToTexture(B.data(), W*3);
    int sb=0,cb=0,ob=0,eb=0; cudaGetRasterStats(&sb,&cb,&ob,&eb);

    printf("mesh path : %6d submitted, %6d backface, %6d offscreen -> %6d drawn\n",
           sa,ca,oa,sa-ca-oa);
    printf("host path : %6d submitted, %6d backface, %6d offscreen -> %6d drawn\n",
           sb,cb,ob,sb-cb-ob);

    int litA=0,litB=0,diff=0;
    for(int i=0;i<W*H;i++){
        bool a=A[i*3]>0, b=B[i*3]>0;
        if(a)litA++; if(b)litB++; if(a!=b)diff++;
    }
    printf("pixels lit: mesh %d, host %d, differing %d (%.3f%%)\n",
           litA,litB,diff, litA?100.0*diff/litA:0.0);

    // pixels must match exactly. the triangle counts are allowed to differ by a
    // hair: the GPU contracts the cull cross-product into FMAs and the host does
    // not, so a degenerate triangle sitting exactly on facing==0 can land on
    // either side. such a triangle covers no pixels, which is why the images
    // still agree bit for bit.
    int dtri = (sa-ca-oa) - (sb-cb-ob); if (dtri < 0) dtri = -dtri;
    bool ok = litA>1000 && diff==0 && dtri <= 1 + (sb-cb-ob)/10000;
    printf("\n%s\n", ok ? "PASS - GPU-resident geometry matches the host path"
                        : "FAIL - mesh path differs from host path");
    cleanupCudaRasterizer();
    return ok?0:1;
}
