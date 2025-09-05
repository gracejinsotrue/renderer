#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "SceneNode.h"
#include "model.h"
#include "tgaimage.h"

class Camera
{
public:
    Vec3f position;
    Vec3f target;
    Vec3f up;
    float fov;

    // Camera(Vec3f pos = Vec3f(0, 0, 3), Vec3f tgt = Vec3f(0, 0, 0), Vec3f u = Vec3f(0, 1, 0))
    //     : position(pos), target(tgt), up(u), fov(-1.0f) {}
    Camera(Vec3f pos = Vec3f(0, 0, 3), Vec3f tgt = Vec3f(0, 0, 0), Vec3f u = Vec3f(0, 1, 0))
        : position(pos), target(tgt), up(u), fov(-0.8f) {}

    void rotate(float yaw, float pitch);
    void move(Vec3f direction, float speed);
    void lookAt(Vec3f eye, Vec3f center, Vec3f up_vec);
};

class Light
{
public:
    Vec3f direction;
    Vec3f color;
    float intensity;

    Light(Vec3f dir = Vec3f(1, 1, 1), Vec3f col = Vec3f(1, 1, 1), float intens = 1.0f)
        : direction(dir), color(col), intensity(intens)
    {
        direction.normalize();
    }
};

class Scene
{
private:
    std::unique_ptr<SceneNode> rootNode;
    std::unordered_map<std::string, std::unique_ptr<Model>> loadedModels;
    SceneNode *selectedNode;
    int nodeCounter; // For generating unique names

public:
    Camera camera;
    Light light;
    TGAImage *background;

    Scene();
    ~Scene();

    // for the model loading and management
    SceneNode *loadModel(const std::string &objPath, const std::string &nodeName = "");
    SceneNode *createEmptyNode(const std::string &nodeName = "");

    // for the scene hierarchy
    SceneNode *getRootNode() const { return rootNode.get(); }
    SceneNode *findNode(const std::string &nodeName);
    bool deleteNode(const std::string &nodeName);

    // for the election
    void selectNode(SceneNode *node);
    void selectNode(const std::string &nodeName);
    void clearSelection();
    SceneNode *getSelectedNode() const { return selectedNode; }

    // for transform updates
    void updateAllTransforms();

    // rendering helpers
    void getAllMeshNodes(std::vector<SceneNode *> &meshNodes);
    void getVisibleMeshNodes(std::vector<SceneNode *> &meshNodes);

    // for background ( i forget if it is flipped 180 right now)
    void loadBackground(const std::string &filename);
    void clearBackground();

    // utility
    void printSceneHierarchy() const;
    void clear();
    int getMeshCount() const;

private:
    std::string generateUniqueName(const std::string &baseName);
    std::string extractModelName(const std::string &filepath);
};

#endif // __SCENE_H__