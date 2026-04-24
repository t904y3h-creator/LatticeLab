#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "Rendering/BaseRenderer.h"

class RendererWGPU : public IRenderer {
public:
    RendererWGPU(World& world, wgpu::TextureFormat surfaceFormat);
    ~RendererWGPU() override = default;

    void drawShot(wgpu::CommandEncoder encoder, wgpu::TextureView targetView, wgpu::TextureView depthView, const World& world) override;

    wgpu::RenderPassEncoder getCurrentPass() { return currentPass; }

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
        glm::mat4 view;
        glm::mat4 projection;
        glm::vec4 lightDir;
        glm::vec4 colorMode;   // x = SpeedColorMode
        glm::vec4 maxSpeedSqr; // x = value
        glm::vec4 maxCount;    // x = value
        glm::vec4 typeColors[119];
    };
    wgpu::Buffer uniformBuffer = nullptr;

    struct GridUniforms {
        float cellSize;
        uint32_t dx;
        uint32_t dy;
        uint32_t dz;
    };
    wgpu::Buffer gridUniformBuffer = nullptr;

    // Vertex buffers
    wgpu::Buffer atomQuadVb = nullptr;
    wgpu::Buffer bondVb = nullptr;
    wgpu::Buffer boxVb = nullptr;
    wgpu::Buffer gridLineVb = nullptr;

    // Pipelines
    wgpu::RenderPipeline atomPipeline = nullptr;
    wgpu::RenderPipeline bondPipeline = nullptr;
    wgpu::RenderPipeline boxPipeline = nullptr;
    wgpu::RenderPipeline gridPipeline = nullptr;

    // Bind group layouts
    wgpu::BindGroupLayout atomBindGroupLayout = nullptr;
    wgpu::BindGroupLayout lineBindGroupLayout = nullptr;
    wgpu::BindGroupLayout gridBindGroupLayout = nullptr;

    // Storage buffers
    // TODO заменить на array<bool>
    wgpu::Buffer sbRadius = nullptr;
    wgpu::Buffer sbSel = nullptr; // array<f32>

    size_t sbCapacity_ = 0;
    size_t bondVbCapacity_ = 0;

    wgpu::BindGroup atomBindGroup = nullptr;

    // Текущий render pass (живёт в течение drawShot)
    wgpu::RenderPassEncoder currentPass = nullptr;

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
    std::vector<float> selectedData_;
    std::vector<float> radii;
    std::vector<float> typeData;
};
