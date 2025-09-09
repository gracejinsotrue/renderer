#include "Scene.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <functional>

// camera
void Camera::rotate(float yaw, float pitch)
{
    Vec3f direction = target - position;
    float radius = direction.norm();

    float cosYaw = cos(yaw), sinYaw = sin(yaw);
    float cosPitch = cos(pitch), sinPitch = sin(pitch);

    direction.x = radius * sinPitch * cosYaw;
    direction.y = radius * cosPitch;
    direction.z = radius * sinPitch * sinYaw;

    target = position + direction;
}

void Camera::move(Vec3f direction, float speed)
{
    Vec3f forward = (target - position).normalize();
    Vec3f right = cross(forward, up).normalize();
    Vec3f realUp = cross(right, forward).normalize();

    Vec3f movement = right * direction.x + realUp * direction.y + forward * direction.z;
    position = position + movement * speed;
    target = target + movement * speed;
}

void Camera::lookAt(Vec3f eye, Vec3f center, Vec3f up_vec)
{
    position = eye;
    target = center;
    up = up_vec;
}

// scene implementation
Scene::Scene() : selectedNode(nullptr), nodeCounter(0), background(nullptr)
{
    rootNode = std::make_unique<SceneNode>("Root", SceneNode::EMPTY);
    std::cout << "Scene created with root node" << std::endl;
}

Scene::~Scene()
{
    clear();
    clearBackground();
}

SceneNode *Scene::loadModel(const std::string &objPath, const std::string &nodeName)
{
    std::cout << "Loading model: " << objPath << std::endl;

    // create unique model if not already loaded
    auto modelIt = loadedModels.find(objPath);
    if (modelIt == loadedModels.end())
    {
        auto model = std::make_unique<Model>(objPath.c_str());
        if (model->nverts() == 0)
        {
            std::cerr << "Failed to load model: " << objPath << std::endl;
            return nullptr;
        }
        loadedModels[objPath] = std::move(model);
        std::cout << "Model loaded with " << loadedModels[objPath]->nverts() << " vertices" << std::endl;
    }

    // generate node name
    std::string finalName = nodeName.empty() ? extractModelName(objPath) : nodeName;
    finalName = generateUniqueName(finalName);

    // create scene node
    auto meshNode = std::make_unique<SceneNode>(finalName, SceneNode::MESH);
    meshNode->attachModel(loadedModels[objPath].get());

    // add to root
    SceneNode *nodePtr = meshNode.get();
    rootNode->addChild(std::move(meshNode));

    std::cout << "Created mesh node: " << finalName << std::endl;
    return nodePtr;
}

SceneNode *Scene::createEmptyNode(const std::string &nodeName)
{
    std::string finalName = nodeName.empty() ? "Empty" : nodeName;
    finalName = generateUniqueName(finalName);

    auto emptyNode = std::make_unique<SceneNode>(finalName, SceneNode::EMPTY);
    SceneNode *nodePtr = emptyNode.get();
    rootNode->addChild(std::move(emptyNode));

    std::cout << "Created empty node: " << finalName << std::endl;
    return nodePtr;
}

SceneNode *Scene::findNode(const std::string &nodeName)
{
    if (rootNode->name == nodeName)
    {
        return rootNode.get();
    }
    return rootNode->findDescendant(nodeName);
}

bool Scene::deleteNode(const std::string &nodeName)
{
    if (nodeName == "Root")
    {
        std::cerr << "Cannot delete root node!" << std::endl;
        return false;
    }

    SceneNode *node = findNode(nodeName);
    if (!node || !node->parent)
    {
        std::cerr << "Node not found or has no parent: " << nodeName << std::endl;
        return false;
    }

    // clear selection if deleting selected node
    if (selectedNode == node)
    {
        selectedNode = nullptr;
    }

    // remove from parent (this will delete the node)
    SceneNode *parent = node->parent;
    parent->removeChild(nodeName);
    delete node; // since removeChild releases ownership

    std::cout << "Deleted node: " << nodeName << std::endl;
    return true;
}

void Scene::selectNode(SceneNode *node)
{
    // clear previous selection
    if (selectedNode)
    {
        selectedNode->selected = false;
    }

    selectedNode = node;
    if (selectedNode)
    {
        selectedNode->selected = true;
        std::cout << "Selected node: " << selectedNode->name << std::endl;
    }
}

void Scene::selectNode(const std::string &nodeName)
{
    SceneNode *node = findNode(nodeName);
    selectNode(node);
}

void Scene::clearSelection()
{
    if (selectedNode)
    {
        selectedNode->selected = false;
        selectedNode = nullptr;
        std::cout << "Selection cleared" << std::endl;
    }
}

void Scene::updateAllTransforms()
{
    rootNode->updateWorldTransform();
}

void Scene::getAllMeshNodes(std::vector<SceneNode *> &meshNodes)
{
    std::function<void(SceneNode *)> collectMeshes = [&](SceneNode *node)
    {
        if (node->hasModel())
        {
            meshNodes.push_back(node);
        }
        for (auto &child : node->children)
        {
            collectMeshes(child.get());
        }
    };

    collectMeshes(rootNode.get());
}

void Scene::getVisibleMeshNodes(std::vector<SceneNode *> &meshNodes)
{
    std::function<void(SceneNode *)> collectVisible = [&](SceneNode *node)
    {
        if (node->hasModel() && node->isVisible())
        {
            meshNodes.push_back(node);
        }
        if (node->isVisible())
        { // only traverse visible branches
            for (auto &child : node->children)
            {
                collectVisible(child.get());
            }
        }
    };

    collectVisible(rootNode.get());
}

void Scene::loadBackground(const std::string &filename)
{
    clearBackground();
    background = new TGAImage();
    if (background->read_tga_file(filename.c_str()))
    {
        std::cout << "Loaded background: " << filename << std::endl;
    }
    else
    {
        delete background;
        background = nullptr;
        std::cerr << "Failed to load background: " << filename << std::endl;
    }
}

void Scene::clearBackground()
{
    if (background)
    {
        delete background;
        background = nullptr;
        std::cout << "Background cleared" << std::endl;
    }
}

void Scene::printSceneHierarchy() const
{
    std::cout << "\n=== Scene Hierarchy ===" << std::endl;
    rootNode->printHierarchy();
    std::cout << "======================" << std::endl;
}

void Scene::clear()
{
    selectedNode = nullptr;
    loadedModels.clear(); // This will delete all models
    rootNode = std::make_unique<SceneNode>("Root", SceneNode::EMPTY);
    nodeCounter = 0;
    std::cout << "Scene cleared" << std::endl;
}

int Scene::getMeshCount() const
{
    std::vector<SceneNode *> meshes;
    const_cast<Scene *>(this)->getAllMeshNodes(const_cast<std::vector<SceneNode *> &>(meshes));
    return meshes.size();
}

std::string Scene::generateUniqueName(const std::string &baseName)
{
    std::string candidate = baseName;
    int suffix = 1;

    while (findNode(candidate) != nullptr)
    {
        candidate = baseName + "_" + std::to_string(suffix);
        suffix++;
    }

    return candidate;
}

std::string Scene::extractModelName(const std::string &filepath)
{
    // extract filename without extension
    std::filesystem::path path(filepath);
    return path.stem().string();
}