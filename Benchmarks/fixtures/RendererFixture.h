#pragma once

#include <cmath>
#include <memory>

#include <benchmark/benchmark.h>

#include "App/Scenes.h"
#include "App/interaction/ToolsManager.h"
#include "App/interaction/picking/PickingSystem.h"
#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/physics/AtomData.h"
#include "Engine/physics/Bond.h"
#include "Rendering/BaseRenderer.h"

template <typename T>
concept IsRenderer = std::derived_from<T, IRenderer>;

template <IsRenderer TRenderer> class RendererFixture : public benchmark::Fixture {
public:
    RendererFixture() : world(Vec3f(160, 160, 160), 1) {}

    void SetUp(benchmark::State& state) override {
        auto& ctx = WGPUContext::instance();
        ctx.initHeadless(800, 600);

        // Offscreen render target
        wgpu::TextureDescriptor colorDesc{};
        colorDesc.size = {800, 600, 1};
        colorDesc.format = ctx.surfaceFormat();
        colorDesc.usage = wgpu::TextureUsage::RenderAttachment;
        colorDesc.mipLevelCount = 1;
        colorDesc.sampleCount = 1;
        colorDesc.dimension = wgpu::TextureDimension::_2D;
        colorTexture_ = ctx.device().createTexture(colorDesc);
        colorTextureView_ = colorTexture_->createView();

        makeGridAtoms(world, static_cast<int>(state.range(0)));
        renderer_ = std::make_unique<TRenderer>(ctx.surfaceFormat(), world);
        renderer_->camera.setScreenSize({800.0f, 600.0f});

        ToolsManager::pickingSystem = new PickingSystem(world, renderer_);
    }

    void TearDown(benchmark::State&) override {
        // Дожидаемся завершения GPU работы
        WGPUContext::instance().device().poll(true, nullptr);
        renderer_.reset();
    }

protected:
    void setCounters(benchmark::State& state) const {
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(world.atomCount()));
    }

    wgpu::raii::Texture colorTexture_;
    wgpu::raii::TextureView colorTextureView_;

    std::unique_ptr<IRenderer> renderer_;
    World world;
    Bond::List bonds_;

private:
    static void makeGridAtoms(World& world, int count) { Scenes::crystal(world, count, AtomData::Type::Z, false); }
};
