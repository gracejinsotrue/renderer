#ifndef __SCENENODE_H__
#define __SCENENODE_H__

#include <vector>
#include <string>
#include <memory>
#include "geometry.h"
#include "model.h"

class Transform
{
public:
    Vec3f position;
    Vec3f rotation; // euler angles in radians
    Vec3f scale;

    Transform() : position(0, 0, 0), rotation(0, 0, 0), scale(1, 1, 1) {}
    Transform(Vec3f pos, Vec3f rot = Vec3f(0, 0, 0), Vec3f scl = Vec3f(1, 1, 1))
        : position(pos), rotation(rot), scale(scl) {}

    Matrix getMatrix() const;
    void interpolate(const Transform &other, float t, Transform &result) const;
};

class SceneNode
{
public:
    std::string name;
    Transform localTransform; // transform relative to parent
    Transform worldTransform; // cached world transform
    bool visible;
    bool selected;

    // for the parent-child hiearchy
    SceneNode *parent;
    std::vector<std::unique_ptr<SceneNode>> children;

    // Optional model data
    Model *model; // raw pointer - scene will manage Model lifetime

    // Node types
    enum NodeType
    {
        EMPTY, // Just for transforms/grouping
        MESH,  // has a model
        LIGHT, // for future light mode
        CAMERA // for future: camera node
    };
    NodeType nodeType;

public:
    SceneNode(const std::string &nodeName, NodeType type = EMPTY);
    ~SceneNode();

    // hierarchy management
    void addChild(std::unique_ptr<SceneNode> child);
    SceneNode *removeChild(const std::string &childName);
    SceneNode *findChild(const std::string &childName);
    SceneNode *findDescendant(const std::string &nodeName); // Recursive search

    // basic transform
    void updateWorldTransform(const Matrix &parentWorld = Matrix::identity());
    Matrix getWorldMatrix() const;
    Matrix getLocalMatrix() const { return localTransform.getMatrix(); }

    // convenience setters
    void setPosition(const Vec3f &pos) { localTransform.position = pos; }
    void setRotation(const Vec3f &rot) { localTransform.rotation = rot; }
    void setScale(const Vec3f &scale) { localTransform.scale = scale; }

    Vec3f getWorldPosition() const { return worldTransform.position; }

    // model management
    void attachModel(Model *modelPtr);
    void detachModel()
    {
        model = nullptr;
        nodeType = EMPTY;
    }
    bool hasModel() const { return model != nullptr; }

    // utility
    void setVisible(bool vis) { visible = vis; }
    bool isVisible() const { return visible; }

    // debug
    void printHierarchy(int depth = 0) const;
    int getChildCount() const { return children.size(); }

private:
    void updateTransformFromMatrix(const Matrix &worldMatrix);
};

#endif // __SCENENODE_H__