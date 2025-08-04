// main.cpp, two pass shadow mapping renderer!

#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>

#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

Model *model = NULL;
const int width = 800;
const int height = 800;

// light source param
Vec3f light_dir(1, 1, 1); // directin to light source
Vec3f eye(0, 0, 3);       // camera pos. i.e. setting y to 0 will be orthogonal to front facing face
Vec3f center(0, 0, 0);    // this is wha cam looks at
Vec3f up(0, 1, 0);        // camera angle

// Global shadow buffer stores depth values from light's perspective
// NOTE: shadowbuffer is declared in our_gl.h and defined in our_gl.cpp lmao
// I didn't declare it here to avoid multiple definition errors and i was stupid and this error caused me grief
const float depth = 255.f; // maximum depth value for normalization (duh)

// FIRST PASS SHADER:
// create the depth buffer from the light's perspective.
// renders the scene from LIGHT'S POV, determines which surfaces are closest to light and casts shadow
struct DepthShader : public IShader
{
    mat<3, 3, float> varying_tri; // this is the screen coord of current triangle

    DepthShader() : varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert)
    {
        // transform vertex from model space -> world space -> light space -> screen space
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));   // read vertex from .obj file
        gl_Vertex = Viewport * Projection * ModelView * gl_Vertex; // transform to screen coordinates
        // just like opengl (this is opengl at home anyways), we store screen coordinates -> depth calculation in fragment shader
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color)
    {
        // calculate the ACTUAL depth of this pixel with barycentric coord
        // https://en.wikipedia.org/wiki/Barycentric_coordinate_system#:~:text=The%20barycentric%20coordinates%20of%20a,point%20is%20inside%20the%20simplex.
        Vec3f p = varying_tri * bar;
        color = TGAColor(255, 255, 255) * (p.z / depth);

        // we store depth in shadow buffer for later use (keep the closest which is represented as smallest depth value)
        int x = (int)p.x;
        int y = (int)p.y;
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            int idx = x + y * width;
            shadowbuffer[idx] = std::min(shadowbuffer[idx], p.z); // Store closest depth
        }

        return false; // DONT DISCRD THE FRAGMENT
    }
};

// SHADER 2: SECOND PASS, MAIN RNDERING WITH SHADOW TESTING
// render from CAMERA perspective and checks each pixel against shadow buffre to see if it is in the shadow
struct ShadowMappingShader : public IShader
{
    mat<4, 4, float> uniform_M;       // Projection*ModelView currnt camera transformation
    mat<4, 4, float> uniform_MIT;     // invert transpose aka (Projection*ModelView).invert_transpose() to transform normals
    mat<4, 4, float> uniform_Mshadow; // transform framebuffer coords to shadowbuffer coords  (camrea coord->lighting coord)
    mat<2, 3, float> varying_uv;      // triangle uv coordinates
    mat<3, 3, float> varying_tri;     // screen coord of triangle verticies before transform

    ShadowMappingShader(Matrix M, Matrix MIT, Matrix MS)
        : uniform_M(M), uniform_MIT(MIT), uniform_Mshadow(MS), varying_uv(), varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert)
    {
        // store uv coord here for textur sampling
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));

        // this is just vertex to screen space, store for later shadow calculation use
        Vec4f gl_Vertex = Viewport * Projection * ModelView * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }
    //

    virtual bool fragment(Vec3f bar, TGAColor &color)
    {
        // currnt pixel -> light coord system
        Vec4f sb_p = uniform_Mshadow * embed<4>(varying_tri * bar);
        sb_p = sb_p / sb_p[3];

        // check if pixel is in shadow by comparing th depth data
        float shadow = 1.0f; // default to being fully lit
        int shadow_x = (int)sb_p[0];
        int shadow_y = (int)sb_p[1];

        if (shadow_x >= 0 && shadow_x < width && shadow_y >= 0 && shadow_y < height)
        {
            int idx = shadow_x + shadow_y * width;
            // Compare current depth with shadow buffer depth
            // If current depth > shadow buffer depth + bias, pixel is in shadow, else lit
            // I ran into z fighting where the planes look like they're overlapping, so I add some proven-to-work coefficient (43.34)
            // https://en.wikipedia.org/wiki/Z-fighting
            shadow = 0.3f + 0.7f * (sb_p[2] <= shadowbuffer[idx] + 43.34f);
        }

        // lighting! standard phong shading
        // https://en.wikipedia.org/wiki/Phong_shading
        Vec2f uv = varying_uv * bar; // interpolatd uv coord
        // normal->texture->world space
        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        // light ->same coord system as normal
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        // reflection vector is R=2N(N-L)-L
        Vec3f r = (n * (n * l * 2.f) - l).normalize(); // reflected light

        // speular is just (R-V)^shininess
        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        // diffuse is just lambert's law of N dot product L
        float diff = std::max(0.f, n * l);

        TGAColor c = model->diffuse(uv); // sampl the base color from diffuse

        // WE JUST CHEESE LIGHTING BY CALCULATING AN ARBITRARY WEIGHTED AVERAGE OF AMBIENT + DIFFUSE +SPECULAR, THIS IS THE MOOOOOOOST
        // STYLISTI CHOICE PART AND I LOVE IT WAHOOOO
        for (int i = 0; i < 3; i++)
        {
            // ambient, diffuse, specular, respectively
            color[i] = std::min<float>(10 + c[i] * shadow * (1.2f * diff + 0.6f * spec), 255);
        }

        return false;
    }
};

int main(int argc, char **argv)
{
    // this part is still in off-the-book mode and using their given 3d model, need to change this next iteration
    if (2 == argc)
    {
        model = new Model(argv[1]);
    }
    else
    {
        model = new Model("obj/african_head.obj");
    }

    light_dir.normalize();

    // Initialize shadow buffer which stores closest depth from light's perspective
    shadowbuffer.resize(width * height);
    std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max()); // Use max, not min

    // FIRST PASS: render from light's perspective to create shadow map
    Matrix M; // matrix store the light's transformation matrix
    {
        std::cout << "Rendering shadow buffer..." << std::endl;
        TGAImage depth(width, height, TGAImage::RGB);
        TGAImage shadowbuffer_img(width, height, TGAImage::GRAYSCALE);

        // clear the shadow buffer with far values (anything closer will overwrite)
        std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max());

        // cam is naer light source pos
        lookat(light_dir, center, up);
        viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        projection(0); // orthographic projection for dirctional light

        // this transformation for converitng camera cord
        M = Viewport * Projection * ModelView;
        // finally render all the triangles w/ depth sahder
        DepthShader depthshader;
        for (int i = 0; i < model->nfaces(); i++)
        {
            Vec4f screen_coords[3];
            for (int j = 0; j < 3; j++)
            {
                screen_coords[j] = depthshader.vertex(i, j);
            }
            triangle(screen_coords, depthshader, depth, shadowbuffer_img);
        }
        // depth buffer viz, comment back in if needed
        //  depth.flip_vertically();
        //  depth.write_tga_file("depth.tga");
    }

    // SECOND PASS: render from camera prspective using shadow buffer
    {
        std::cout << "Rendering main frame with shadows..." << std::endl;
        TGAImage frame(width, height, TGAImage::RGB);
        TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);

        // cam position is back at normal viewing pos
        lookat(eye, center, up);
        viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        projection(-1.f / (eye - center).norm()); // Perspective projection

        // now i create the transformation matrix that converts framebuffer coords to shadowbuffer coords
        // This is: M * inverse(Viewport * Projection * ModelView)
        Matrix current_transform = Viewport * Projection * ModelView;

        // this part was my skill issue, we are supposed to do M * inverse(current_transform) but
        // I forgot .invert() is a thing, so i did defintion of inverse matrix
        // aka adjugate/determinant = inverse haha skill issue
        Matrix shadow_transform = M * (current_transform.adjugate() / current_transform.det());

        // fianlly create main shader with all transformation matrices
        ShadowMappingShader shader(
            ModelView,                                   // lgiht direction transform
            (Projection * ModelView).invert_transpose(), // normal transform
            shadow_transform);                           // shadow coord converstion
        // render w/ shadow mapping
        for (int i = 0; i < model->nfaces(); i++)
        {
            Vec4f screen_coords[3];
            for (int j = 0; j < 3; j++)
            {
                screen_coords[j] = shader.vertex(i, j);
            }
            triangle(screen_coords, shader, frame, zbuffer);
        }
        // save
        frame.flip_vertically();
        zbuffer.flip_vertically();
        frame.write_tga_file("output.tga"); // this is the ifnal rendered image
        // zbuffer.write_tga_file("zbuffer.tga");
    }

    delete model;
    return 0;
}