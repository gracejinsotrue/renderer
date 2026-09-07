#include "UI.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdlrenderer2.h"

#include "Engine.h"

namespace
{
// An exponential average with a per-frame weight rather than a per-second one.
// The readout only has to be legible, and tying it to elapsed time would make
// the smoothing itself jump around whenever the frame time did.
void smooth(float &acc, float sample)
{
    acc = (acc <= 0.f) ? sample : acc * 0.9f + sample * 0.1f;
}
} // namespace

UI::UI()
    : active(false), use_gl(false), window(nullptr), renderer(nullptr),
      fps_smooth(0.f), upload_ms_smooth(0.f), bin_ms_smooth(0.f),
      raster_ms_smooth(0.f)
{
    snprintf(env_path, sizeof(env_path), "environment.hdr");
    snprintf(model_path, sizeof(model_path), "assets/bunny/bunny.obj");
}

UI::~UI() { shutdown(); }

bool UI::init(SDL_Window *win, SDL_GLContext gl_context, SDL_Renderer *rend)
{
    if (active) return true;
    if (!win || (!gl_context && !rend)) return false;

    window = win;
    renderer = rend;
    use_gl = (gl_context != nullptr);

    IMGUI_CHECKVERSION();
    if (!ImGui::CreateContext()) return false;
    ImGui::StyleColorsDark();

    // No imgui.ini. It would land wherever the engine happened to be started
    // from and quietly restore a layout from a different session.
    ImGui::GetIO().IniFilename = nullptr;

    bool ok = false;
    if (use_gl)
    {
        ok = ImGui_ImplSDL2_InitForOpenGL(window, gl_context) &&
             ImGui_ImplOpenGL3_Init(nullptr);
    }
    else
    {
        ok = ImGui_ImplSDL2_InitForSDLRenderer(window, renderer) &&
             ImGui_ImplSDLRenderer2_Init(renderer);
    }

    if (!ok)
    {
        ImGui::DestroyContext();
        return false;
    }

    active = true;
    return true;
}

void UI::shutdown()
{
    if (!active) return;
    if (use_gl)
    {
        ImGui_ImplOpenGL3_Shutdown();
    }
    else
    {
        ImGui_ImplSDLRenderer2_Shutdown();
    }
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    active = false;
}

void UI::handleEvent(const SDL_Event &e)
{
    if (active) ImGui_ImplSDL2_ProcessEvent(&e);
}

bool UI::wantsMouse() const
{
    return active && ImGui::GetIO().WantCaptureMouse;
}

bool UI::wantsKeyboard() const
{
    return active && ImGui::GetIO().WantCaptureKeyboard;
}

void UI::buildFrameStats(Engine &engine)
{
    smooth(fps_smooth, engine.getFPS());

    int submitted = 0, culled_back = 0, culled_off = 0, bin_entries = 0;
    cudaGetRasterStats(&submitted, &culled_back, &culled_off, &bin_entries);

    float upload = 0.f, bin = 0.f, raster = 0.f;
    cudaGetKernelTimings(&upload, &bin, &raster);
    smooth(upload_ms_smooth, upload);
    smooth(bin_ms_smooth, bin);
    smooth(raster_ms_smooth, raster);

    ImGui::Text("%.1f FPS", fps_smooth);
    ImGui::Separator();

    // These are already measured every frame with CUDA events; before this
    // panel the only way to see them was tests/bin/profile_frame.
    ImGui::Text("GPU, last flush");
    ImGui::Text("  setup + upload  %6.3f ms", upload_ms_smooth);
    ImGui::Text("  bin             %6.3f ms", bin_ms_smooth);
    ImGui::Text("  raster + shade  %6.3f ms", raster_ms_smooth);
    ImGui::Separator();

    int drawn = submitted - culled_back - culled_off;
    ImGui::Text("Triangles");
    ImGui::Text("  offered         %8d", submitted);
    ImGui::Text("  rasterized      %8d", drawn < 0 ? 0 : drawn);
    ImGui::Text("  backface culled %8d", culled_back);
    ImGui::Text("  offscreen       %8d", culled_off);
    ImGui::Text("  bin entries     %8d", bin_entries);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("(triangle, tile) pairs, not an overflow. A triangle\n"
                          "covering nine tiles is nine entries, so this is\n"
                          "normally larger than the triangle count.");
    ImGui::Text("  shaded pixels   %8d", cudaGetShadeCount());
}

void UI::buildDisplay(Engine &engine)
{
    float exposure = engine.getExposure();
    // Logarithmic: the useful range for an HDR map runs from about 0.05 to 2,
    // and a linear slider spends most of its travel above that.
    if (ImGui::SliderFloat("Exposure", &exposure, 0.01f, 64.f, "%.3f",
                           ImGuiSliderFlags_Logarithmic))
        engine.setExposure(exposure);
    ImGui::SameLine();
    if (ImGui::SmallButton("1.0##exp")) engine.setExposure(1.0f);

    int ss = engine.getSSAA();
    const char *ss_names[] = {"1x", "2x", "4x"};
    int ss_index = (ss >= 4) ? 2 : (ss >= 2 ? 1 : 0);
    if (ImGui::Combo("Supersampling", &ss_index, ss_names, 3))
        engine.setSSAA(1 << ss_index);

    bool deferred = cudaGetDeferredShading() != 0;
    if (ImGui::Checkbox("Deferred shading", &deferred))
        cudaSetDeferredShading(deferred ? 1 : 0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off shades inside the depth loop, so cost follows\n"
                          "overdraw rather than screen area.");

    float bias = engine.getShadowBias();
    if (ImGui::SliderFloat("Shadow bias", &bias, 0.f, 12.f, "%.2f"))
        engine.setShadowBias(bias);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A fraction of the 0..255 depth range, not a world\n"
                          "distance. Too low gives acne, too high detaches\n"
                          "contact shadows.");

    if (ImGui::Button("Capture frame")) engine.captureFrame("output.tga");
    ImGui::SameLine();
    if (ImGui::Button("Reset camera")) engine.resetCamera();
}

void UI::buildLighting(Engine &engine)
{
    ImGui::InputText("##envpath", env_path, sizeof(env_path));
    ImGui::SameLine();
    if (ImGui::Button("Load .hdr")) engine.loadEnvironment(env_path);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) engine.getScene().clearEnvironment();

    bool has_env = engine.getScene().environment != nullptr;
    ImGui::TextDisabled(has_env ? "environment loaded (lights the scene)"
                                : "no environment: ambient is flat grey");

    float ibl = engine.getIBLIntensity();
    if (ImGui::SliderFloat("IBL intensity", &ibl, 0.f, 4.f, "%.2f"))
        engine.setIBLIntensity(ibl);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("How much of the environment's irradiance reaches\n"
                          "the shading. No effect without one loaded.");

    ImGui::SeparatorText("Directional light");

    Light &light = engine.getScene().light;

    // Direction as a unit vector, renormalised after any edit: the shading
    // treats it as one, and an unnormalised drag would read as a brightness
    // change rather than a direction change.
    float dir[3] = {light.direction.x, light.direction.y, light.direction.z};
    if (ImGui::SliderFloat3("Direction", dir, -1.f, 1.f, "%.2f"))
    {
        light.direction = Vec3f(dir[0], dir[1], dir[2]);
        if (light.direction.norm() > 1e-4f) light.direction.normalize();
    }

    float intensity = light.intensity;
    if (ImGui::SliderFloat("Intensity", &intensity, 0.f, 4.f, "%.2f"))
        light.intensity = intensity;

    float col[3] = {light.color.x, light.color.y, light.color.z};
    if (ImGui::ColorEdit3("Colour", col))
        light.color = Vec3f(col[0], col[1], col[2]);

    ImGui::SeparatorText("Ambient occlusion");

    bool on = engine.isSSAOEnabled();
    if (ImGui::Checkbox("Enabled", &on)) engine.toggleSSAO();

    float ao = engine.getSSAOIntensity();
    if (ImGui::SliderFloat("Strength", &ao, 0.f, 2.f, "%.2f"))
        engine.setSSAOIntensity(ao);

    const char *views[] = {"Shaded", "Occlusion term", "Eye-space normals",
                           "Depth"};
    int view = engine.getSSAODebug();
    if (view < 0 || view > 3) view = 0;
    if (ImGui::Combo("View", &view, views, 4)) engine.setSSAODebug(view);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The debug views carry measurements rather than\n"
                          "light, so they bypass the tone curve.");
}

void UI::buildScene(Engine &engine)
{
    Scene &scene = engine.getScene();

    ImGui::InputText("##modelpath", model_path, sizeof(model_path));
    ImGui::SameLine();
    if (ImGui::Button("Load model")) engine.loadModel(model_path);

    std::vector<SceneNode *> nodes;
    scene.getAllMeshNodes(nodes);
    SceneNode *selected = scene.getSelectedNode();

    ImGui::Text("%d mesh node(s)", (int)nodes.size());
    if (ImGui::BeginListBox("##nodes", ImVec2(-FLT_MIN, 6 * ImGui::GetTextLineHeightWithSpacing())))
    {
        for (SceneNode *n : nodes)
        {
            if (ImGui::Selectable(n->name.c_str(), n == selected))
                scene.selectNode(n);
        }
        ImGui::EndListBox();
    }

    if (!selected)
    {
        ImGui::TextDisabled("nothing selected");
        return;
    }

    ImGui::SeparatorText(selected->name.c_str());

    // Local, not world: this is what the node owns. Its parent's transform is
    // composed on top by Scene::updateAllTransforms.
    Transform &t = selected->localTransform;
    ImGui::DragFloat3("Position", &t.position.x, 0.01f);

    float deg[3] = {t.rotation.x * 57.2957795f, t.rotation.y * 57.2957795f,
                    t.rotation.z * 57.2957795f};
    if (ImGui::DragFloat3("Rotation", deg, 0.5f, -360.f, 360.f, "%.1f deg"))
        t.rotation = Vec3f(deg[0] / 57.2957795f, deg[1] / 57.2957795f,
                           deg[2] / 57.2957795f);

    ImGui::DragFloat3("Scale", &t.scale.x, 0.01f, 0.001f, 100.f);

    if (ImGui::Button("Duplicate")) engine.duplicateSelectedObject();
    ImGui::SameLine();
    if (ImGui::Button("Delete")) engine.deleteSelectedObject();
}

void UI::build(Engine &engine)
{
    if (!active) return;

    if (use_gl)
        ImGui_ImplOpenGL3_NewFrame();
    else
        ImGui_ImplSDLRenderer2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 640), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Renderer"))
    {
        if (ImGui::CollapsingHeader("Frame", ImGuiTreeNodeFlags_DefaultOpen))
            buildFrameStats(engine);
        if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
            buildDisplay(engine);
        if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
            buildLighting(engine);
        if (ImGui::CollapsingHeader("Scene"))
            buildScene(engine);
    }
    ImGui::End();

    ImGui::Render();
}

void UI::render()
{
    if (!active) return;
    ImDrawData *data = ImGui::GetDrawData();
    if (!data) return;

    if (use_gl)
        ImGui_ImplOpenGL3_RenderDrawData(data);
    else
        ImGui_ImplSDLRenderer2_RenderDrawData(data, renderer);
}
