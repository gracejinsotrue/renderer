// The camera transform the raster kernels are fed. Building these on the host
// costs a handful of 4x4 multiplies per mesh per frame; the result is uploaded
// to the GPU, which is where every per-vertex and per-pixel use of it happens.
#include <cmath>
#include "transform.h"

Matrix ModelView;
Matrix Viewport;
Matrix Projection;

void viewport(int x, int y, int w, int h)
{
    Viewport = Matrix::identity();
    Viewport[0][3] = x + w / 2.f;
    Viewport[1][3] = y + h / 2.f;
    Viewport[2][3] = 255.f / 2.f;
    Viewport[0][0] = w / 2.f;
    Viewport[1][1] = h / 2.f;
    Viewport[2][2] = 255.f / 2.f;
}

void projection(float coeff)
{
    Projection = Matrix::identity();
    Projection[3][2] = coeff;
    // Projection[2][3] = -1.0f / coeff;
}

void lookat(Vec3f eye, Vec3f center, Vec3f up)
{
    Vec3f z = (eye - center).normalize();
    Vec3f x = cross(up, z).normalize();
    Vec3f y = cross(z, x).normalize();
    ModelView = Matrix::identity();
    for (int i = 0; i < 3; i++)
    {
        ModelView[0][i] = x[i];
        ModelView[1][i] = y[i];
        ModelView[2][i] = z[i];
    }
    // the translation is -R*center, not -center. those agree only when center
    // is the origin; any other target lands geometry at the wrong camera-space
    // depth, giving negative w.
    ModelView[0][3] = -(x * center);
    ModelView[1][3] = -(y * center);
    ModelView[2][3] = -(z * center);
}
