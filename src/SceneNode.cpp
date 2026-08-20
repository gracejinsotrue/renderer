#include "SceneNode.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// transform implementation
Matrix Transform::getMatrix() const
{
    Matrix translation = Matrix::identity();
    translation[0][3] = position.x;
    translation[1][3] = position.y;
    translation[2][3] = position.z;

    // rotation matrices in XYZ order
    Matrix rotX = Matrix::identity();
    float cosX = cos(rotation.x), sinX = sin(rotation.x);
    rotX[1][1] = cosX;
    rotX[1][2] = -sinX;
    rotX[2][1] = sinX;
    rotX[2][2] = cosX;

    Matrix rotY = Matrix::identity();
    float cosY = cos(rotation.y), sinY = sin(rotation.y);
    rotY[0][0] = cosY;
    rotY[0][2] = sinY;
    rotY[2][0] = -sinY;
    rotY[2][2] = cosY;

    Matrix rotZ = Matrix::identity();
    float cosZ = cos(rotation.z), sinZ = sin(rotation.z);
    rotZ[0][0] = cosZ;
    rotZ[0][1] = -sinZ;
    rotZ[1][0] = sinZ;
    rotZ[1][1] = cosZ;

    Matrix scaling = Matrix::identity();
    scaling[0][0] = scale.x;
    scaling[1][1] = scale.y;
    scaling[2][2] = scale.z;

    // basic order is: Translation * Rotation * Scale
    return translation * rotZ * rotY * rotX * scaling;
}

void Transform::interpolate(const Transform &other, float t, Transform &result) const
{
    result.position = position + (other.position - position) * t;
    result.rotation = rotation + (other.rotation - rotation) * t;
    result.scale = scale + (other.scale - scale) * t;
}

// sceneNode implementation
SceneNode::SceneNode(const std::string &nodeName, NodeType type)
    : name(nodeName), visible(true), selected(false), parent(nullptr),
      model(nullptr), nodeType(type)
{

    std::cout << "Created SceneNode: " << name << " (type: " << type << ")" << std::endl;
}

SceneNode::~SceneNode()
{
    // don't delete model here - the scene manages Model lifetime
    std::cout << "Destroying SceneNode: " << name << std::endl;
}

void SceneNode::addChild(std::unique_ptr<SceneNode> child)
{
    if (!child)
        return;

    child->parent = this;
    std::cout << "Adding child '" << child->name << "' to '" << name << "'" << std::endl;
    children.push_back(std::move(child));
}

SceneNode *SceneNode::removeChild(const std::string &childName)
{
    auto it = std::find_if(children.begin(), children.end(),
                           [&childName](const std::unique_ptr<SceneNode> &child)
                           {
                               return child->name == childName;
                           });

    if (it != children.end())
    {
        SceneNode *rawPtr = it->get();
        rawPtr->parent = nullptr;
        it->release(); // release ownership without deleting
        children.erase(it);
        std::cout << "Removed child '" << childName << "' from '" << name << "'" << std::endl;
        return rawPtr;
    }
    return nullptr;
}

SceneNode *SceneNode::findChild(const std::string &childName)
{
    auto it = std::find_if(children.begin(), children.end(),
                           [&childName](const std::unique_ptr<SceneNode> &child)
                           {
                               return child->name == childName;
                           });

    return (it != children.end()) ? it->get() : nullptr;
}
// find children
SceneNode *SceneNode::findDescendant(const std::string &nodeName)
{
    // check direct children first
    for (auto &child : children)
    {
        if (child->name == nodeName)
        {
            return child.get();
        }
    }

    // rcursively search in children
    for (auto &child : children)
    {
        SceneNode *found = child->findDescendant(nodeName);
        if (found)
            return found;
    }

    return nullptr;
}

void SceneNode::updateWorldTransform(const Matrix &parentWorld)
{
    Matrix localMatrix = getLocalMatrix();
    Matrix worldMatrix = parentWorld * localMatrix;

    // extract world transform from matrix
    updateTransformFromMatrix(worldMatrix);

    // update all children
    for (auto &child : children)
    {
        child->updateWorldTransform(worldMatrix);
    }
}

Matrix SceneNode::getWorldMatrix() const
{
    return worldTransform.getMatrix();
}

void SceneNode::updateTransformFromMatrix(const Matrix &worldMatrix)
{
    // extract position
    worldTransform.position.x = worldMatrix[0][3];
    worldTransform.position.y = worldMatrix[1][3];
    worldTransform.position.z = worldMatrix[2][3];

    // extract scale
    Vec3f scaleX(worldMatrix[0][0], worldMatrix[1][0], worldMatrix[2][0]);
    Vec3f scaleY(worldMatrix[0][1], worldMatrix[1][1], worldMatrix[2][1]);
    Vec3f scaleZ(worldMatrix[0][2], worldMatrix[1][2], worldMatrix[2][2]);

    worldTransform.scale.x = scaleX.norm();
    worldTransform.scale.y = scaleY.norm();
    worldTransform.scale.z = scaleZ.norm();

    // extract rotation, assume no skew right now
    if (worldTransform.scale.x > 0.0001f && worldTransform.scale.y > 0.0001f && worldTransform.scale.z > 0.0001f)
    {
        // normalize rotation matrix
        Matrix rotMatrix = Matrix::identity();
        rotMatrix[0][0] = worldMatrix[0][0] / worldTransform.scale.x;
        rotMatrix[0][1] = worldMatrix[0][1] / worldTransform.scale.y;
        rotMatrix[0][2] = worldMatrix[0][2] / worldTransform.scale.z;
        rotMatrix[1][0] = worldMatrix[1][0] / worldTransform.scale.x;
        rotMatrix[1][1] = worldMatrix[1][1] / worldTransform.scale.y;
        rotMatrix[1][2] = worldMatrix[1][2] / worldTransform.scale.z;
        rotMatrix[2][0] = worldMatrix[2][0] / worldTransform.scale.x;
        rotMatrix[2][1] = worldMatrix[2][1] / worldTransform.scale.y;
        rotMatrix[2][2] = worldMatrix[2][2] / worldTransform.scale.z;

        // extract Euler angles in XYZ order
        worldTransform.rotation.y = asin(-rotMatrix[2][0]);

        if (cos(worldTransform.rotation.y) > 0.0001f)
        {
            worldTransform.rotation.x = atan2(rotMatrix[2][1], rotMatrix[2][2]);
            worldTransform.rotation.z = atan2(rotMatrix[1][0], rotMatrix[0][0]);
        }
        else
        {
            worldTransform.rotation.x = atan2(-rotMatrix[1][2], rotMatrix[1][1]);
            worldTransform.rotation.z = 0;
        }
    }
}

void SceneNode::attachModel(Model *modelPtr)
{
    model = modelPtr;
    nodeType = MESH;
    std::cout << "Attached model to node '" << name << "'" << std::endl;
}

void SceneNode::printHierarchy(int depth) const
{
    for (int i = 0; i < depth; i++)
        std::cout << "  ";

    std::cout << "- " << name;
    if (hasModel())
        std::cout << " [MESH]";
    if (!visible)
        std::cout << " [HIDDEN]";
    if (selected)
        std::cout << " [SELECTED]";
    std::cout << std::endl;

    for (const auto &child : children)
    {
        child->printHierarchy(depth + 1);
    }
}