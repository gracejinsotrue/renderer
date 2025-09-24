#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "model.h"

Model::Model(const char *filename) : verts_(), faces_(), norms_(), uv_(), diffusemap_(), normalmap_(), specularmap_(), hasBackup_(false)
{
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail())
        return;
    std::string line;
    while (!in.eof())
    {
        std::getline(in, line);
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v "))
        {
            iss >> trash;
            Vec3f v;
            for (int i = 0; i < 3; i++)
                iss >> v[i];
            verts_.push_back(v);
        }
        else if (!line.compare(0, 3, "vn "))
        {
            iss >> trash >> trash;
            Vec3f n;
            for (int i = 0; i < 3; i++)
                iss >> n[i];
            norms_.push_back(n);
        }
        else if (!line.compare(0, 3, "vt "))
        {
            iss >> trash >> trash;
            Vec2f uv;
            for (int i = 0; i < 2; i++)
                iss >> uv[i];
            uv_.push_back(uv);
        }
        else if (!line.compare(0, 2, "f "))
        {
            std::vector<Vec3i> f;
            Vec3i tmp;
            iss >> trash;
            while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2])
            {
                for (int i = 0; i < 3; i++)
                    tmp[i]--; // in wavefront obj all indices start at 1, not zero
                f.push_back(tmp);
            }
            faces_.push_back(f);
        }
    }
    std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << " vt# " << uv_.size() << " vn# " << norms_.size() << std::endl;
    load_texture(filename, "_diffuse.tga", diffusemap_);
    load_texture(filename, "_nm.tga", normalmap_);
    load_texture(filename, "_spec.tga", specularmap_);

    // Automatically backup original vertices for animation
    backupOriginalVertices();
}

Model::~Model() {}

int Model::nverts()
{
    return (int)verts_.size();
}

int Model::nfaces()
{
    return (int)faces_.size();
}

std::vector<int> Model::face(int idx)
{
    std::vector<int> face;
    for (int i = 0; i < (int)faces_[idx].size(); i++)
        face.push_back(faces_[idx][i][0]);
    return face;
}

Vec3f Model::vert(int i)
{
    return verts_[i];
}

Vec3f Model::vert(int iface, int nthvert)
{
    return verts_[faces_[iface][nthvert][0]];
}

void Model::load_texture(std::string filename, const char *suffix, TGAImage &img)
{
    std::string texfile(filename);
    size_t dot = texfile.find_last_of(".");
    if (dot != std::string::npos)
    {
        texfile = texfile.substr(0, dot) + std::string(suffix);
        std::cerr << "texture file " << texfile << " loading " << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
        img.flip_vertically();
    }
}

TGAColor Model::diffuse(Vec2f uvf)
{
    Vec2i uv(uvf[0] * diffusemap_.get_width(), uvf[1] * diffusemap_.get_height());
    return diffusemap_.get(uv[0], uv[1]);
}

Vec3f Model::normal(Vec2f uvf)
{
    Vec2i uv(uvf[0] * normalmap_.get_width(), uvf[1] * normalmap_.get_height());
    TGAColor c = normalmap_.get(uv[0], uv[1]);
    Vec3f res;
    for (int i = 0; i < 3; i++)
        res[2 - i] = (float)c[i] / 255.f * 2.f - 1.f;
    return res;
}

Vec2f Model::uv(int iface, int nthvert)
{
    return uv_[faces_[iface][nthvert][1]];
}

float Model::specular(Vec2f uvf)
{
    Vec2i uv(uvf[0] * specularmap_.get_width(), uvf[1] * specularmap_.get_height());
    return specularmap_.get(uv[0], uv[1])[0] / 1.f;
}

Vec3f Model::normal(int iface, int nthvert)
{
    int idx = faces_[iface][nthvert][2];
    return norms_[idx].normalize();
}

// EXISTING ANIMATION METHODS
void Model::setVertex(int i, const Vec3f &newPos)
{
    if (i >= 0 && i < verts_.size())
    {
        verts_[i] = newPos;
    }
}

Vec3f *Model::getVertexData()
{
    if (verts_.empty())
        return nullptr;
    return &verts_[0];
}

void Model::updateVertex(int index, const Vec3f &offset)
{
    if (index >= 0 && index < verts_.size())
    {
        verts_[index] = verts_[index] + offset;
    }
}

void Model::resetVertices()
{
    if (hasBackup_)
    {
        verts_ = originalVerts_;
    }
}

void Model::backupOriginalVertices()
{
    originalVerts_ = verts_;
    hasBackup_ = true;
    std::cout << "Backed up " << originalVerts_.size() << " original vertices for animation" << std::endl;
}

void Model::restoreOriginalVertices()
{
    if (hasBackup_)
    {
        verts_ = originalVerts_;
        std::cout << "Restored original vertices" << std::endl;
    }
}

// blend shape
void Model::addBlendShape(const std::string &name, const std::vector<Vec3f> &targetVertices)
{
    if (targetVertices.size() != verts_.size())
    {
        std::cerr << "blend shape vertex count mismatch!" << std::endl;
        return;
    }

    blendShapes[name] = targetVertices;
    blendWeights[name] = 0.0f;
    std::cout << "Added blend shape: " << name << std::endl;
}

void Model::setBlendWeight(const std::string &shapeName, float weight)
{
    if (blendShapes.find(shapeName) != blendShapes.end())
    {
        blendWeights[shapeName] = std::max(0.0f, std::min(1.0f, weight));
        std::cout << "Set " << shapeName << " weight to " << weight << std::endl;
    }
}

void Model::applyBlendShapes()
{
    if (!hasBackup_)
    {
        std::cout << "No backup vertices - creating backup now" << std::endl;
        backupOriginalVertices();
    }

    // Start with original vertices
    verts_ = originalVerts_;

    // Apply each blend shape with its weight
    for (const auto &shape : blendShapes)
    {
        const std::string &name = shape.first;
        const std::vector<Vec3f> &targetVerts = shape.second;
        float weight = blendWeights[name];

        if (weight > 0.0f)
        {
            for (size_t i = 0; i < verts_.size(); i++)
            {
                Vec3f offset = (targetVerts[i] - originalVerts_[i]) * weight;
                verts_[i] = verts_[i] + offset;
            }
        }
    }

    std::cout << "Applied blend shapes to " << verts_.size() << " vertices" << std::endl;
}
// TODO: clear up this these tests now tha we know this works

void Model::createTestBlendShapes()
{
    if (verts_.empty())
    {
        std::cerr << "No vertices to create blend shapes from!" << std::endl;
        return;
    }

    std::cout << "Creating test blend shapes for " << verts_.size() << " vertices..." << std::endl;

    // ereate "expand" blend shape - push all vertices outward from center
    std::vector<Vec3f> expandVerts = verts_;
    Vec3f center(0, 0, 0);

    // calculate center
    for (const Vec3f &v : verts_)
    {
        center = center + v;
    }
    center = center / float(verts_.size());

    // push vertices outward by n%
    for (size_t i = 0; i < expandVerts.size(); i++)
    {
        Vec3f direction = (expandVerts[i] - center).normalize();
        expandVerts[i] = expandVerts[i] + direction * 0.1f;
    }
    addBlendShape("expand", expandVerts);

    // create "squash" blend shape - compress Y axis
    std::vector<Vec3f> squashVerts = verts_;
    for (Vec3f &v : squashVerts)
    {
        v.y *= 0.8f; // Compress vertically
    }
    addBlendShape("squash", squashVerts);

    // create "twist" blend shape - rotate upper vertices
    std::vector<Vec3f> twistVerts = verts_;
    for (size_t i = 0; i < twistVerts.size(); i++)
    {
        if (twistVerts[i].y > center.y)
        { // Only affect upper vertices
            float angle = 0.3f;
            float x = twistVerts[i].x;
            float z = twistVerts[i].z;
            twistVerts[i].x = x * cos(angle) - z * sin(angle);
            twistVerts[i].z = x * sin(angle) + z * cos(angle);
        }
    }
    addBlendShape("twist", twistVerts);

    std::cout << "Created 3 test blend shapes: expand, squash, twist" << std::endl;
}

void Model::listBlendShapes() const
{
    if (blendShapes.empty())
    {
        std::cout << "No blend shapes saved." << std::endl;
        return;
    }

    std::cout << "\n=== SAVED BLEND SHAPES ===" << std::endl;
    int index = 1;
    for (const auto &shape : blendShapes)
    {
        float weight = 0.0f;
        auto weightIt = blendWeights.find(shape.first);
        if (weightIt != blendWeights.end())
        {
            weight = weightIt->second;
        }
        std::cout << "  " << index << ". " << shape.first
                  << " (current weight: " << (weight * 100) << "%)" << std::endl;
        index++;
    }
    std::cout << "==========================" << std::endl;
}

std::vector<std::string> Model::getBlendShapeNames() const
{
    std::vector<std::string> names;
    for (const auto &shape : blendShapes)
    {
        names.push_back(shape.first);
    }
    return names;
}

bool Model::hasBlendShape(const std::string &name) const
{
    return blendShapes.find(name) != blendShapes.end();
}

void Model::clearAllBlendWeights()
{
    for (auto &weight : blendWeights)
    {
        weight.second = 0.0f;
    }
    applyBlendShapes();
    std::cout << "Cleared all blend shape weights - returned to neutral expression" << std::endl;
}

void Model::setExpressionByName(const std::string &name, float intensity)
{
    // clear all weights first
    clearAllBlendWeights();

    // set the target expression
    if (hasBlendShape(name))
    {
        setBlendWeight(name, intensity);
        applyBlendShapes();
        std::cout << "Applied expression '" << name << "' at " << (intensity * 100) << "% intensity" << std::endl;
    }
    else
    {
        std::cout << "Expression '" << name << "' not found!" << std::endl;
        listBlendShapes();
    }
}

void Model::blendBetweenExpressions(const std::string &from, const std::string &to, float t)
{
    if (!hasBlendShape(from) || !hasBlendShape(to))
    {
        std::cout << "One or both expressions not found: '" << from << "', '" << to << "'" << std::endl;
        return;
    }

    // clear all weights
    for (auto &weight : blendWeights)
    {
        weight.second = 0.0f;
    }

    // lerp blend between the two expressions
    float fromWeight = 1.0f - t;
    float toWeight = t;

    setBlendWeight(from, fromWeight);
    setBlendWeight(to, toWeight);
    applyBlendShapes();

    std::cout << "Blending " << (fromWeight * 100) << "% '" << from
              << "' with " << (toWeight * 100) << "% '" << to << "'" << std::endl;
}

void Model::saveCurrentStateAsBlendShape(const std::string &name)
{
    addBlendShape(name, verts_);
    std::cout << "Saved current vertex state as blend shape: '" << name << "'" << std::endl;
}