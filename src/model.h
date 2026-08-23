#ifndef __MODEL_H__
#define __MODEL_H__
#include <vector>
#include <string>
#include <map>
#include "geometry.h"
#include "tgaimage.h"

class Model
{
private:
    enum TextureFilterMode
    {
        FILTER_POINT,
        FILTER_LINEAR
    };
    static TextureFilterMode textureFilterMode_;
    std::vector<Vec3f> verts_;
    std::vector<std::vector<Vec3i>> faces_; // this Vec3i means vertex/uv/normal
    std::vector<Vec3f> norms_;
    std::vector<Vec2f> uv_;
    TGAImage diffusemap_;
    TGAImage normalmap_;
    TGAImage specularmap_;
    void load_texture(std::string filename, const char *suffix, TGAImage &img);

public:
    Model(const char *filename);
    ~Model();
    int nverts();
    int nfaces();
    Vec3f normal(int iface, int nthvert);
    Vec3f normal(Vec2f uv);
    Vec3f vert(int i);
    Vec3f vert(int iface, int nthvert);
    Vec2f uv(int iface, int nthvert);
    TGAColor diffuse(Vec2f uv);
    float specular(Vec2f uv);
    std::vector<int> face(int idx);

    static void setLinearTextureFiltering(bool enabled);
    static bool usesLinearTextureFiltering();

    // raw maps, for uploading to the GPU as textures
    TGAImage &diffuseMap() { return diffusemap_; }
    TGAImage &normalMap() { return normalmap_; }
    TGAImage &specularMap() { return specularmap_; }

    // flat positions, for uploading the mesh to the GPU. const because a
    // loaded mesh is immutable: nothing edits geometry after the file is read.
    const Vec3f *getVertexData() const;
};
#endif //__MODEL_H__
