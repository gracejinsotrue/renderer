#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "model.h"

Model::TextureFilterMode Model::textureFilterMode_ = Model::FILTER_LINEAR;

static float clamp01(float v)
{
    return std::max(0.0f, std::min(1.0f, v));
}

static TGAColor sampleBilinear(TGAImage &img, Vec2f uvf)
{
    int w = img.get_width();
    int h = img.get_height();
    if (w <= 0 || h <= 0)
        return TGAColor();

    float u = clamp01(uvf[0]);
    float v = clamp01(uvf[1]);

    // Match normalized linear texture lookup: uv=0/1 clamps to the texture
    // edge, while interior samples blend around texel centres.
    float fx = u * w - 0.5f;
    float fy = v * h - 0.5f;
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    x0 = std::max(0, std::min(w - 1, x0));
    y0 = std::max(0, std::min(h - 1, y0));
    x1 = std::max(0, std::min(w - 1, x1));
    y1 = std::max(0, std::min(h - 1, y1));

    TGAColor c00 = img.get(x0, y0);
    TGAColor c10 = img.get(x1, y0);
    TGAColor c01 = img.get(x0, y1);
    TGAColor c11 = img.get(x1, y1);
    TGAColor out;
    out.bytespp = c00.bytespp;
    for (int i = 0; i < out.bytespp; i++)
    {
        float top = c00[i] * (1.0f - tx) + c10[i] * tx;
        float bot = c01[i] * (1.0f - tx) + c11[i] * tx;
        out[i] = (unsigned char)std::min(255.0f, std::max(0.0f, top * (1.0f - ty) + bot * ty));
    }
    return out;
}

static TGAColor sampleNearest(TGAImage &img, Vec2f uvf)
{
    int w = img.get_width();
    int h = img.get_height();
    if (w <= 0 || h <= 0)
        return TGAColor();
    float u = clamp01(uvf[0]);
    float v = clamp01(uvf[1]);
    int x = std::max(0, std::min(w - 1, (int)(u * w)));
    int y = std::max(0, std::min(h - 1, (int)(v * h)));
    return img.get(x, y);
}

static TGAColor sampleTexture(TGAImage &img, Vec2f uvf)
{
    return Model::usesLinearTextureFiltering() ? sampleBilinear(img, uvf)
                                               : sampleNearest(img, uvf);
}

Model::Model(const char *filename) : verts_(), faces_(), norms_(), uv_(), diffusemap_(), normalmap_(), specularmap_()
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
    load_texture(filename, "_diffuse.tga", diffusemap_);
    load_texture(filename, "_nm.tga", normalmap_);
    load_texture(filename, "_spec.tga", specularmap_);
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
        // a model without a normal or specular map is normal, so only
        // complain when one that exists fails to parse
        if (!img.read_tga_file(texfile.c_str()))
            std::cerr << "  no " << suffix << " map for " << filename << std::endl;
        img.flip_vertically();
    }
}

TGAColor Model::diffuse(Vec2f uvf)
{
    return sampleTexture(diffusemap_, uvf);
}

Vec3f Model::normal(Vec2f uvf)
{
    TGAColor c = sampleTexture(normalmap_, uvf);
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
    return sampleTexture(specularmap_, uvf)[0] / 1.f;
}

void Model::setLinearTextureFiltering(bool enabled)
{
    textureFilterMode_ = enabled ? FILTER_LINEAR : FILTER_POINT;
}

bool Model::usesLinearTextureFiltering()
{
    return textureFilterMode_ == FILTER_LINEAR;
}

Vec3f Model::normal(int iface, int nthvert)
{
    int idx = faces_[iface][nthvert][2];
    return norms_[idx].normalize();
}

const Vec3f *Model::getVertexData() const
{
    if (verts_.empty())
        return nullptr;
    return &verts_[0];
}

