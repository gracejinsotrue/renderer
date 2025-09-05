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
    std::vector<Vec3f> verts_;
    std::vector<std::vector<Vec3i>> faces_; // this Vec3i means vertex/uv/normal
    std::vector<Vec3f> norms_;
    std::vector<Vec2f> uv_;
    TGAImage diffusemap_;
    TGAImage normalmap_;
    TGAImage specularmap_;
    void load_texture(std::string filename, const char *suffix, TGAImage &img);

    // Backup/restore for animation
    std::vector<Vec3f> originalVerts_; // Backup of original vertices
    bool hasBackup_;

    // Blend shape system
    std::map<std::string, std::vector<Vec3f>> blendShapes;
    std::map<std::string, float> blendWeights;

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

    void setVertex(int i, const Vec3f &newPos);
    Vec3f *getVertexData();
    const std::vector<Vec3f> &getVertices() const { return verts_; }
    void updateVertex(int index, const Vec3f &offset);
    void resetVertices(); // Reset to original positions

    void backupOriginalVertices();
    void restoreOriginalVertices();

    // Blend shape methods
    void addBlendShape(const std::string &name, const std::vector<Vec3f> &targetVertices);
    void setBlendWeight(const std::string &shapeName, float weight);
    void applyBlendShapes();

    // Test helper - create procedural deformations
    void createTestBlendShapes();

    // Enhanced blend shape management
    void listBlendShapes() const;
    std::vector<std::string> getBlendShapeNames() const;
    bool hasBlendShape(const std::string &name) const;
    void clearAllBlendWeights();
    void setExpressionByName(const std::string &name, float intensity = 1.0f);

    // Animation and interpolation
    void blendBetweenExpressions(const std::string &from, const std::string &to, float t);
    void saveCurrentStateAsBlendShape(const std::string &name);
};
#endif //__MODEL_H__