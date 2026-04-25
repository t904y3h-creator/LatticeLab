#include "AppActions.h"

#include "App/AppSignals.h"
#include "App/Scenes.h"
#include "App/capture/CaptureController.h"
#include "App/interaction/ToolsManager.h"
#include "App/save_system/AppStateIO.h"
#include "Engine/Simulation.h"
#include "Engine/gpu/WGPUContext.h"
#include "GUI/interface/UiState.h"
#include "Rendering/2d/Renderer2DWGPU.h"
#include "Rendering/3d/Renderer3DWGPU.h"

namespace {
    void shiftAtoms(GpuAtomBuffers& atomBuffers, const Vec3f& delta) {
        // TODO переписать на шейдер
        const size_t count = atomBuffers.countAtoms();
        if (count == 0) {
            return;
        }

        std::vector<Vec3f> positions(count);
        atomBuffers.downloadPositions(positions);
        for (Vec3f& p : positions) {
            p += delta;
        }
        atomBuffers.uploadPositions(positions);
    }

    void applyResizeBox(Simulation& simulation, std::unique_ptr<IRenderer>& renderer, const Vec3f& newSize) {
        const Vec3f oldSize = simulation.world().getWorldSize();
        const Vec3f delta = (newSize - oldSize) * 0.5f;

        shiftAtoms(simulation.world().getAtomBuffers(), delta);
        renderer->camera.move(delta.xy());
        renderer->camera.move3D(delta);
        simulation.world().setWorldSize(newSize);
    }
}

namespace AppActions {
    void Handler::trackIOPanel(CaptureController& captureController, UiState& uiState, Simulation& simulation,
                               std::unique_ptr<IRenderer>& renderer) {
        track(AppSignals::UI::SaveSimulation.connect(
            [&](std::string_view path) { AppStateIO::save(captureController, uiState.scenePreviewRect, simulation, *renderer, path); }));
        track(AppSignals::UI::LoadSimulation.connect([&](std::string_view path) {
            AppStateIO::load(simulation, *renderer, path);
            ToolsManager::resetInteractionState();
        }));
        track(AppSignals::UI::ResizeBox.connect([&](const Vec3f& newSize) { applyResizeBox(simulation, renderer, newSize); }));
        track(AppSignals::UI::ClearSimulation.connect([&]() {
            simulation.world().clearAtoms();
            ToolsManager::resetInteractionState();
        }));
        track(AppSignals::UI::CreateGas.connect([&](int atomCount, AtomData::Type atomType, bool is3D, float density) {
            simulation.world().clearAtoms();
            ToolsManager::resetInteractionState();
            Scenes::randomGas(simulation.world(), atomCount, atomType, is3D, 6.0f, 6.0f, density);
        }));
        track(AppSignals::UI::CreateCrystal.connect([&](int axisCount, AtomData::Type atomType, bool is3D) {
            simulation.world().clearAtoms();
            ToolsManager::resetInteractionState();
            Scenes::crystal(simulation.world(), axisCount, atomType, is3D);
        }));
    }

    void Handler::trackToolsPanel(Simulation& simulation, std::unique_ptr<IRenderer>& renderer) {
        track(AppSignals::UI::SetRender.connect([&](RendererType type) {
            std::unique_ptr<IRenderer> newRenderer;
            switch (type) {
            case RendererType::Renderer2D:
                newRenderer = std::make_unique<Renderer2DWGPU>(WGPUContext::instance().surfaceFormat(), simulation.world());
                break;
            case RendererType::Renderer3D:
                newRenderer = std::make_unique<Renderer3DWGPU>(WGPUContext::instance().surfaceFormat(), simulation.world());
                break;
            }

            if (newRenderer) {
                ToolsManager::resetInteractionState();
                newRenderer->drawGrid = renderer->drawGrid;
                newRenderer->drawBonds = renderer->drawBonds;
                newRenderer->speedColorMode = renderer->speedColorMode;
                newRenderer->speedGradientMax = renderer->speedGradientMax;
                newRenderer->camera.setScreenSize(renderer->camera.getScreenSize());
                renderer = std::move(newRenderer);
            }
        }));

        track(AppSignals::UI::SetCameraMode.connect([&](Camera::Mode mode) { renderer->camera.setMode(mode); }));
    }

    void Handler::trackSettingsPanel(GLFWwindow* window) {
        track(AppSignals::UI::ExitApplication.connect([window]() { glfwSetWindowShouldClose(window, GLFW_TRUE); }));
    }

    void Handler::trackKeyboard(Simulation& simulation) {
        track(AppSignals::Keyboard::StepPhysics.connect([&]() { simulation.step(); }));
    }

    void Handler::trackSimControlPanel(Simulation& simulation) {
        track(AppSignals::UI::StepPhysics.connect([&]() { simulation.step(); }));
    }

    Handler::Handler(GLFWwindow* window, CaptureController& captureController, Simulation& simulation, std::unique_ptr<IRenderer>& renderer,
                     UiState& uiState) {
        trackIOPanel(captureController, uiState, simulation, renderer);
        trackToolsPanel(simulation, renderer);
        trackSettingsPanel(window);
        trackSimControlPanel(simulation);
        trackKeyboard(simulation);
    }
}
