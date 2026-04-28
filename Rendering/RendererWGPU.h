#pragma once

#include <vector>

#include <glm/glm.hpp>
#include <webgpu/webgpu-raii.hpp>

#include "Engine/math/Vec3.h"
#include "Rendering/BaseRenderer.h"

class RendererWGPU : public IRenderer {
public:
    RendererWGPU(wgpu::TextureFormat surfaceFormat, World& world);
    ~RendererWGPU() override = default;

    void drawShot(wgpu::CommandEncoder encoder, wgpu::TextureView targetView, wgpu::TextureView depthView, const World& world) override;

    wgpu::raii::RenderPassEncoder& getCurrentPass() { return currentPass; }

protected:
    virtual void updateMatrices() = 0;
    virtual glm::vec3 getLightDir() = 0;
    virtual bool useLighting() = 0;

    glm::mat4 projection{1.f};
    glm::mat4 view{1.f};

    void initAtomPipeline(std::string_view atomWGSL);
    void initGridPipeline(std::string_view gridWGSL);
    void initBoxPipeline(std::string_view boxWGSL);
    void initBondPipeline(std::string_view bondWGSL);

private:
    struct SceneUniforms {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 projection;
        alignas(16) glm::vec4 lightDir;
        uint32_t colorMode;
        float maxSpeedSqr;
        uint32_t maxCount;
        float _pad;
        alignas(16) glm::vec4 typeColors[119];
    };
    wgpu::raii::Buffer uniformBuffer;

    struct GridUniforms {
        alignas(16) Vec3u gridSize;
        float cellSize;
    };
    wgpu::raii::Buffer gridUniformBuffer;

    // Vertex buffers
    wgpu::raii::Buffer atomQuadVb;
    wgpu::raii::Buffer bondVb;
    wgpu::raii::Buffer boxVb;
    wgpu::raii::Buffer gridLineVb;

    // Pipelines
    wgpu::raii::RenderPipeline atomPipeline;
    wgpu::raii::RenderPipeline bondPipeline;
    wgpu::raii::RenderPipeline boxPipeline;
    wgpu::raii::RenderPipeline gridPipeline;

    // Bind group layouts
    wgpu::raii::BindGroupLayout atomBindGroupLayout;
    wgpu::raii::BindGroupLayout lineBindGroupLayout;
    wgpu::raii::BindGroupLayout gridBindGroupLayout;

    // Storage buffers
    // TODO заменить на array<bool>
    wgpu::raii::Buffer sbRadius;
    wgpu::raii::Buffer sbSel; // array<f32>

    size_t sbCapacity_ = 0;
    size_t bondVbCapacity_ = 0;

    wgpu::raii::BindGroup atomBindGroup;

    // Текущий render pass (живёт в течение drawShot)
    wgpu::raii::RenderPassEncoder currentPass;

    wgpu::TextureFormat surfaceFormat;

    void initAtomColors();
    void initAtomQuadBuffer();
    void initBoxBuffer();
    void initBondBuffer();
    void initGridLineBuffer();
    void initLinePipeline(wgpu::RenderPipeline& outPipeline, std::string_view wgsl);

    // Helpers
    void rebuildBindGroup(const World& world);
    void uploadRadii(const World& world);

    // Draw
    void drawAtomsImpl(const World& world);
    // void drawBondsImpl(const AtomStorage& atoms, const Bond::List& bonds);
    void drawBoxImpl(const Vec3f& worldSize);
    void drawGridImpl(const World& world);

    // Data
    std::vector<glm::vec4> typeColorsData;
    std::vector<std::uint32_t> selectedData_;
    std::vector<float> radii;
    std::vector<float> typeData;
};
