#include "Application.h"

#include <cmath>
#include <cstdlib>

#include <imgui_impl_wgpu.h>

#include "App/AppActions.h"
#include "App/CreateWindow.h"
#include "App/Scenes.h"
#include "App/UserSettings.h"
#include "App/interaction/ToolsManager.h"
#include "Engine/Simulation.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/metrics/Profiler.h"
#include "GUI/interface/interface.h"
#include "GUI/io/keyboard/Keyboard.h"
#include "GUI/io/manager/EventManager.h"
#include "Rendering/2d/Renderer2DWGPU.h"
#include "capture/CaptureActions.h"
#include "capture/CaptureController.h"
#include "debug/CreateDebugPanels.h"
#include "debug/DebugRuntime.h"

using Clock = std::chrono::high_resolution_clock;

constexpr int FPS = 60;
constexpr int LPS = 20;

int Application::run() {
    GLFWwindow* window = createWindow();
    if (!window) {
        return EXIT_FAILURE;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    WGPUContext::instance().init(window, width, height);

    // инициализация систем
    World world(Vec3f(50, 50, 6), 0);
    Scenes::crystal(world, 400, AtomData::Type::Z, false);

    Simulation simulation(world);
    CaptureController captureController;

    std::unique_ptr<IRenderer> renderer = std::make_unique<Renderer2DWGPU>(WGPUContext::instance().surfaceFormat(), world);

    Interface appInterface(window, simulation, renderer, captureController);
    AppActions::Handler appActions(window, captureController, simulation, renderer, appInterface.state());
    CaptureActions::Handler captureActions(captureController);

    if (appInterface.init() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    EventManager::init(window, renderer, appInterface);
    ToolsManager::init(window, simulation, renderer, appInterface);

    const DebugViews debugViews = createDebugViews(appInterface.debugPanel);

    // загрузка пользовательских настроек
    const UserSettings userSettings = UserSettingsIO::load();
    captureController.setSettings(userSettings.captureSettings);
    captureController.setOutputDirectory(userSettings.captureOutputDirectory);
    appInterface.setScenesDirectory(userSettings.scenesDirectory);
    renderer->drawGrid = userSettings.rendererDrawGrid;
    renderer->drawBonds = userSettings.rendererDrawBonds;
    renderer->speedColorMode = userSettings.rendererSpeedColorMode;
    renderer->speedGradientMax = userSettings.rendererSpeedGradientMax;
    world.setLJEnabled(userSettings.simulationLJEnabled);
    world.setCoulombEnabled(userSettings.simulationCoulombEnabled);
    appInterface.state().simulationSpeed = 100.0f;
    appInterface.state().pause = true;

    auto startTime = Clock::now();
    double renderAccum = 0.0;
    double physicsAccum = 0.0;
    double logAccum = 0.0;

    constexpr double renderInterval = 1.0 / FPS;
    constexpr double logInterval = 1.0 / LPS;

    renderer->camera.resetView();

    while (!glfwWindowShouldClose(window)) {
        Profiler::instance().beginFrame();

        const auto currentTime = Clock::now();
        const float deltaTime = std::chrono::duration<float>(currentTime - startTime).count();
        startTime = currentTime;

        physicsAccum += deltaTime;
        renderAccum += deltaTime;
        logAccum += deltaTime;

        EventManager::poll();
        EventManager::frame(deltaTime);
        captureController.update(deltaTime);
        captureController.syncUiState(appInterface.state());
        captureController.handleToggleShortcut();

        UiState& uiState = appInterface.state();
        WGPUContext& ctx = WGPUContext::instance();

        static std::array<wgpu::CommandBuffer, 2> submitBatch;
        uint32_t submitCount = 0;

        // Физика
        const double physicsInterval = 1.0 / uiState.simulationSpeed;
        if (physicsAccum >= physicsInterval) {
            physicsAccum = 0.f;

            if (!uiState.pause) {
                wgpu::CommandEncoderDescriptor desc;
                desc.label = wgpu::StringView("PhysicsEncoder");
                wgpu::CommandEncoder physEnc = ctx.device().createCommandEncoder(desc);
                simulation.step(physEnc);
                submitBatch[submitCount++] = physEnc.finish({});
            }
        }

        // Рендер
        if (renderAccum >= renderInterval) {
            PROFILE_SCOPE("Application::RenderFrame");
            renderAccum -= renderInterval;

            uiState.simStep = simulation.getSimStep();
            appInterface.update();
            refreshAtomDebugViews(debugViews, simulation);

            wgpu::SurfaceTexture surfaceTex;
            ctx.surface().getCurrentTexture(&surfaceTex);
            wgpu::Texture surfaceTexture(surfaceTex.texture);
            wgpu::TextureView renderTarget = captureController.acquireRenderTarget(surfaceTexture);

            wgpu::CommandEncoderDescriptor desc;
            desc.label = wgpu::StringView("RenderEncoder");
            wgpu::CommandEncoder renderEnc = ctx.device().createCommandEncoder(desc);

            renderer->drawShot(renderEnc, renderTarget, ctx.depthView(), world);
            ToolsManager::pickingSystem->getOverlay().draw();

            ImGui::Render();
            auto* wgpuRenderer = static_cast<RendererWGPU*>(renderer.get());
            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), wgpuRenderer->getCurrentPass());
            wgpuRenderer->getCurrentPass().end();
            wgpuRenderer->getCurrentPass() = nullptr;

            submitBatch[submitCount++] = renderEnc.finish({});

            ctx.queue().submit(submitCount, submitBatch.data());
            submitCount = 0;

            captureController.onFrameRendered(surfaceTexture);
            ctx.present();
            ctx.processEvents();
        }

        // физика без рендера в этом кадре
        if (submitCount > 0) {
            ctx.queue().submit(submitCount, submitBatch.data());
            submitCount = 0;
        }

        submitBatch.fill(nullptr);

        Profiler::instance().endFrame();

        if (logAccum >= logInterval) {
            logAccum -= logInterval;
            refreshSimulationDebugViews(debugViews, simulation);
        }
    }

    captureController.stop();
    UserSettingsIO::save(UserSettings{
        .captureOutputDirectory = captureController.outputDirectory(),
        .scenesDirectory = appInterface.scenesDirectory(),
        .captureSettings = captureController.settings(),
        .rendererDrawGrid = renderer->drawGrid,
        .rendererDrawBonds = renderer->drawBonds,
        .rendererSpeedColorMode = renderer->speedColorMode,
        .rendererSpeedGradientMax = renderer->speedGradientMax,
        .simulationBondFormationEnabled = false,
        .simulationLJEnabled = world.isLJEnabled(),
        .simulationCoulombEnabled = world.isCoulombEnabled(),
    });
    appInterface.shutdown();

    return 0;
}
