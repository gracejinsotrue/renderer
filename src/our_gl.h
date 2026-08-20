#ifndef __OUR_GL_H__
#define __OUR_GL_H__

#include "tgaimage.h"
#include "geometry.h"
#include <vector>

extern Matrix ModelView;
extern Matrix Viewport;
extern Matrix Projection;

// Global shadow buffer for shadow mapping, plus its own dimensions. index it
// with these, not with the `width`/`height` globals from main.cpp, which are
// unrelated to the size this buffer is allocated at.
extern std::vector<float> shadowbuffer;
extern int shadowbuffer_w, shadowbuffer_h;

// allocates the shadow buffer and records its size. clears to -inf: the map
// keeps the surface nearest the light, and depth grows toward the viewer here,
// so nearest means largest.
void resizeShadowBuffer(int w, int h);
void clearShadowBuffer();

void viewport(int x, int y, int w, int h);
void projection(float coeff = 0.f); // coeff = -1/c
void lookat(Vec3f eye, Vec3f center, Vec3f up);

struct IShader
{
    virtual ~IShader();
    virtual Vec4f vertex(int iface, int nthvert) = 0;
    virtual bool fragment(Vec3f screen_bar, Vec3f persp_bar, TGAColor &color) = 0;
};

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, TGAImage &zbuffer);

#endif //__OUR_GL_H__