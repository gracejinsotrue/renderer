#ifndef __SHADERS_H__
#define __SHADERS_H__

#include "our_gl.h"
#include "model.h"
#include "geometry.h"

// Global variables that shaders need. the shadow buffer's dimensions come
// from our_gl.h, not from main.cpp's width/height.
extern Model *model;
extern Vec3f light_dir;

// how far behind the shadow map's stored depth a fragment may sit before it
// counts as occluded. depth spans 0..255 regardless of world scale, so this is
// a fraction of that range, not a world-space distance. matches the CUDA path.
static const float SHADOW_BIAS = 2.0f;

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

    virtual bool fragment(Vec3f bar, Vec3f, TGAColor &color)
    {
        Vec3f p = varying_tri * bar;
        color = TGAColor(255, 255, 255) * (p.z / 255.f);

        int x = (int)p.x;
        int y = (int)p.y;
        if (x >= 0 && x < shadowbuffer_w && y >= 0 && y < shadowbuffer_h)
        {
            int idx = x + y * shadowbuffer_w;
            // max, not min: depth grows toward the viewer, so the surface
            // closest to the light is the one with the LARGEST z. taking the
            // minimum stores the far side and inverts every shadow test.
            shadowbuffer[idx] = std::max(shadowbuffer[idx], p.z);
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
    Vec3f uniform_light_color;
    float uniform_light_intensity;
    mat<2, 3, float> varying_uv;
    mat<3, 3, float> varying_tri;

    ShadowMappingShader(Matrix M, Matrix MIT, Matrix MS, Vec3f light_color, float light_intensity)
        : uniform_M(M), uniform_MIT(MIT), uniform_Mshadow(MS),
          uniform_light_color(light_color),
          uniform_light_intensity(std::max(0.0f, light_intensity)),
          varying_uv(), varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert)
    {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = Viewport * Projection * ModelView * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f screen_bar, Vec3f persp_bar, TGAColor &color)
    {
        Vec4f sb_p = uniform_Mshadow * embed<4>(varying_tri * screen_bar);
        sb_p = sb_p / sb_p[3];

        float shadow = 1.0f;
        int shadow_x = (int)sb_p[0];
        int shadow_y = (int)sb_p[1];

        if (shadow_x >= 0 && shadow_x < shadowbuffer_w &&
            shadow_y >= 0 && shadow_y < shadowbuffer_h)
        {
            int idx = shadow_x + shadow_y * shadowbuffer_w;
            // lit when this fragment is the one the light can see, i.e. its
            // depth reaches the nearest depth recorded for that texel
            shadow = 0.3f + 0.7f * (sb_p[2] + SHADOW_BIAS >= shadowbuffer[idx]);
        }

        Vec2f uv = varying_uv * persp_bar;

        if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
        {
            color = TGAColor(255, 0, 255); // Magenta for missing UVs
            return false;
        }

        // embed<4> fills the new component with 1 by default, which is right
        // for a position and wrong for a direction: it adds the matrix's
        // translation column. both of these are directions.
        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv), 0.f)).normalize();
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir, 0.f)).normalize();
        Vec3f r = (n * (n * l * 2.f) - l).normalize();

        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n * l);

        TGAColor c = model->diffuse(uv);
        if (c[0] == 0 && c[1] == 0 && c[2] == 0)
        {
            color = TGAColor(0, 255, 0); // Green for black texture samples
            return false;
        }

        float lit = shadow * uniform_light_intensity * (0.8f * diff + 0.3f * spec);
        color[0] = std::min<float>(20 + c[0] * lit * std::max(0.0f, uniform_light_color.z), 255);
        color[1] = std::min<float>(20 + c[1] * lit * std::max(0.0f, uniform_light_color.y), 255);
        color[2] = std::min<float>(20 + c[2] * lit * std::max(0.0f, uniform_light_color.x), 255);

        return false;
    }
};

#endif // __SHADERS_H__