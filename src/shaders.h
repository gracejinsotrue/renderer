#ifndef __SHADERS_H__
#define __SHADERS_H__

#include "our_gl.h"
#include "model.h"
#include "geometry.h"

// Global variables that shaders need
extern Model *model;
extern Vec3f light_dir;
extern const int width;
extern const int height;

// FIRST PASS SHADER: Depth buffer creation from light's perspective
struct DepthShader : public IShader
{
    mat<3, 3, float> varying_tri;

    DepthShader() : varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert)
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        gl_Vertex = Viewport * Projection * ModelView * gl_Vertex;
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color)
    {
        Vec3f p = varying_tri * bar;
        color = TGAColor(255, 255, 255) * (p.z / 255.f);

        int x = (int)p.x;
        int y = (int)p.y;
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            int idx = x + y * width;
            shadowbuffer[idx] = std::min(shadowbuffer[idx], p.z);
        }

        return false;
    }
};

// SECOND PASS SHADER: Main rendering with shadow mapping
struct ShadowMappingShader : public IShader
{
    mat<4, 4, float> uniform_M;
    mat<4, 4, float> uniform_MIT;
    mat<4, 4, float> uniform_Mshadow;
    mat<2, 3, float> varying_uv;
    mat<3, 3, float> varying_tri;

    ShadowMappingShader(Matrix M, Matrix MIT, Matrix MS)
        : uniform_M(M), uniform_MIT(MIT), uniform_Mshadow(MS), varying_uv(), varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert)
    {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = Viewport * Projection * ModelView * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color)
    {
        Vec4f sb_p = uniform_Mshadow * embed<4>(varying_tri * bar);
        sb_p = sb_p / sb_p[3];

        float shadow = 1.0f;
        int shadow_x = (int)sb_p[0];
        int shadow_y = (int)sb_p[1];

        if (shadow_x >= 0 && shadow_x < width && shadow_y >= 0 && shadow_y < height)
        {
            int idx = shadow_x + shadow_y * width;
            shadow = 0.3f + 0.7f * (sb_p[2] <= shadowbuffer[idx] + 43.34f);
        }

        Vec2f uv = varying_uv * bar;

        if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
        {
            color = TGAColor(255, 0, 255); // Magenta for missing UVs
            return false;
        }

        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        Vec3f r = (n * (n * l * 2.f) - l).normalize();

        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n * l);

        TGAColor c = model->diffuse(uv);
        if (c[0] == 0 && c[1] == 0 && c[2] == 0)
        {
            color = TGAColor(0, 255, 0); // Green for black texture samples
            return false;
        }

        for (int i = 0; i < 3; i++)
        {
            color[i] = std::min<float>(20 + c[i] * shadow * (0.8f * diff + 0.3f * spec), 255);
        }

        return false;
    }
};

#endif // __SHADERS_H__