// headless capture of an arbitrary set of models, for eyeballing a scene
// without a window. usage: capture_scene out.tga model.obj [model.obj ...]
//
// Camera, light and pass settings come from the environment rather than more
// positional arguments, so a figure in the README can be reproduced by pasting
// one line. Everything defaults to what the interactive engine starts with.
//
//   ORBIT=yaw[,pitch]  orbit the camera, radians          (default 0)
//   ZOOM=delta         add to the camera distance          (default 0, dist 3)
//   SCALE=f            scale every loaded node             (default 1)
//   PLACE=n:x,y,z[,s];...  move named nodes, optional per-node scale
//   LIGHT=x,y,z        directional light, world space      (default 1,1,1)
//   INTENSITY=f        light intensity                     (default 1)
//   SSAA=1|2|4         supersampling factor                (default 2)
//   NO_SSAO=1          turn the occlusion pass off         (default on)
//   SSAO_RADIUS=f      occlusion radius, world units
//   SSAO_INTENSITY=f   occlusion strength, 0..1
//   SSAO_DEBUG=0..4    render the AO term / normals / depth instead
//   SHADOW_BIAS=f      shadow comparison bias, in 0..255 depth units
//                      (default 2; negative renders the shadow debug view)
//   EYE=x,y,z          camera position, overrides ORBIT/ZOOM
//   LOOK=x,y,z         what the camera points at         (default the origin)
//   ENV=path.hdr       equirectangular environment: backdrop and IBL
//   EXPOSURE=f         linear multiplier before the tone curve (default 1)
//   IBL=f              how much environment irradiance reaches the shading
//   METALLIC=f         0 dielectric, 1 conductor           (default 0)
//   ROUGHNESS=f        perceptual, 0..1                    (default 0.5)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "Engine.h"
#include "compat.h"

// the pre-port Engine.h pulls in shaders.h, which declares these extern
Model *model = NULL;
Vec3f light_dir(1, 1, 1);

// reads "a,b,c" into out; returns how many components were present
static int envVec(const char *name, float *out, int n)
{
    const char *s = getenv(name);
    if (!s) return 0;
    int got = 0;
    while (got < n && *s)
    {
        out[got++] = (float)atof(s);
        const char *comma = strchr(s, ',');
        if (!comma) break;
        s = comma + 1;
    }
    return got;
}

int main(int argc, char **argv)
{
    if (argc < 3) { printf("usage: %s out.tga model.obj [model.obj ...]\n", argv[0]); return 2; }
    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    if (!engine.init()) { printf("init failed\n"); return 1; }

    for (int i = 2; i < argc; i++)
        if (!engine.loadModel(argv[i])) printf("could not load %s\n", argv[i]);

    // The projection is tinyrenderer's weak-perspective coefficient, which
    // carries no field-of-view term: a point on the plane through the model
    // centre maps x_ndc = x_eye, so a model of radius 1 lands exactly on the
    // frustum edge no matter where the camera stands, and moving the camera
    // does not reframe it. Since the setup kernel culls whole triangles against
    // those planes without clipping them, a silhouette sitting on the boundary
    // comes out as a sawtooth. Scaling the nodes is what actually reframes.
    if (getenv("SCALE"))
    {
        float s = (float)atof(getenv("SCALE"));
        std::vector<SceneNode *> meshes;
        engine.getScene().getAllMeshNodes(meshes);
        for (SceneNode *n : meshes)
            n->setScale(Vec3f(s, s, s));
    }

    // Node placement, so one capture can hold several models rather than
    // stacking them all on the origin. Names are the ones Scene::loadModel
    // derives from the file stem, which is what printSceneHierarchy shows.
    if (const char *place = getenv("PLACE"))
    {
        std::vector<SceneNode *> meshes;
        engine.getScene().getAllMeshNodes(meshes);
        for (const char *e = place; e && *e; )
        {
            const char *colon = strchr(e, ':');
            const char *semi = strchr(e, ';');
            if (!colon || (semi && colon > semi)) break;
            std::string name(e, colon - e);
            float t[4] = {0.f, 0.f, 0.f, 0.f};
            // the 4th field is optional: a per-node scale, so a group shot can
            // size the models without also shrinking the ground they stand on
            int got = sscanf(colon + 1, "%f,%f,%f,%f", &t[0], &t[1], &t[2], &t[3]);
            bool hit = false;
            for (SceneNode *n : meshes)
                if (n->name == name)
                {
                    n->localTransform.position = Vec3f(t[0], t[1], t[2]);
                    if (got >= 4) n->localTransform.scale = Vec3f(t[3], t[3], t[3]);
                    hit = true;
                }
            if (!hit) printf("PLACE: no node named %s\n", name.c_str());
            e = semi ? semi + 1 : NULL;
        }
    }

    if (getenv("SSAA")) engine.setSSAA(atoi(getenv("SSAA")));

    if (getenv("NO_SSAO") && engine.isSSAOEnabled()) engine.toggleSSAO();
    if (getenv("SSAO_RADIUS")) engine.setSSAORadius((float)atof(getenv("SSAO_RADIUS")));
    if (getenv("SSAO_INTENSITY")) engine.setSSAOIntensity((float)atof(getenv("SSAO_INTENSITY")));
    if (getenv("SSAO_DEBUG")) engine.setSSAODebug(atoi(getenv("SSAO_DEBUG")));
    if (getenv("SHADOW_BIAS")) engine.setShadowBias((float)atof(getenv("SHADOW_BIAS")));

    float l[3];
    if (envVec("LIGHT", l, 3) == 3)
    {
        engine.getScene().light.direction = Vec3f(l[0], l[1], l[2]).normalize();
        light_dir = engine.getScene().light.direction;
    }
    if (getenv("INTENSITY"))
        engine.getScene().light.intensity = (float)atof(getenv("INTENSITY"));

    // Loaded before the material settings so a figure can pair a surface with
    // the environment lighting it, which is the only way a metal shows
    // anything at all.
    if (const char *env = getenv("ENV"))
    {
        engine.loadEnvironment(env);
        if (!engine.getScene().environment) printf("could not load %s\n", env);
    }
    if (getenv("EXPOSURE")) engine.setExposure((float)atof(getenv("EXPOSURE")));
    if (getenv("IBL")) engine.setIBLIntensity((float)atof(getenv("IBL")));
    if (getenv("METALLIC")) engine.setMetallic((float)atof(getenv("METALLIC")));
    if (getenv("ROUGHNESS")) engine.setRoughness((float)atof(getenv("ROUGHNESS")));

    if (getenv("ZOOM")) engine.zoomCamera((float)atof(getenv("ZOOM")));

    float o[2] = {0.f, 0.f};
    if (envVec("ORBIT", o, 2) > 0) engine.orbitCamera(o[0], o[1]);

    // Explicit camera, applied last so it wins over ORBIT and ZOOM. The orbit
    // controls always point at the origin from outside, which is no use for a
    // scene you have to stand inside: Sponza is a closed box seen from any
    // exterior angle, and its roof is solid, so the only view of the arcade is
    // from within it. Engine::update() does not recompute the camera, so
    // writing position and target here holds through render().
    float eye[3], look[3];
    bool haveEye = envVec("EYE", eye, 3) == 3;
    bool haveLook = envVec("LOOK", look, 3) == 3;
    if (haveEye || haveLook)
    {
        if (haveEye) engine.getScene().camera.position = Vec3f(eye[0], eye[1], eye[2]);
        if (haveLook) engine.getScene().camera.target = Vec3f(look[0], look[1], look[2]);
    }

    engine.update();
    engine.render();
    engine.captureFrame(argv[1]);
    int sub=0, cb=0, co=0, bins=0;
    cudaGetRasterStats(&sub, &cb, &co, &bins);
    printf("STATS submitted=%d culled_back=%d culled_off=%d bin_entries=%d\n",
           sub, cb, co, bins);
    printf("wrote %s\n", argv[1]);
    return 0;
}
