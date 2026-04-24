#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Rendering/BaseRenderer.h"

class RendererWGPU : public IRenderer {
public:
    RendererWGPU(World& simbox, wgpu::Device device, wgpu::TextureFormat surfaceFormat, const GpuAtomBuffers& atomBuffers);
    ~RendererWGPU() override = default;

    void drawShot(wgpu::TextureView targetView, wgpu::TextureView depthView, const AtomStorage& atoms, const Bond::List& bonds,
                  const World& box, const GpuAtomBuffers& atomBuffers) override;
    void endFrame() override;

    wgpu::RenderPassEncoder getCurrentPass() { return currentPass; }

protected:
    virtual void updateMatrices() = 0;
    virtual glm::vec3 getLightDir() = 0;
    virtual bool useLighting() = 0;

    glm::mat4 projection{1.f};
    glm::mat4 view{1.f};

    wgpu::Device device = nullptr;

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

    // Pipelines (аналог bgfx::ProgramHandle)
    wgpu::RenderPipeline atomPipeline = nullptr;
    wgpu::RenderPipeline bondPipeline = nullptr;
    wgpu::RenderPipeline boxPipeline = nullptr;
    wgpu::RenderPipeline gridPipeline = nullptr;

    // Bind group layouts
    wgpu::BindGroupLayout atomBindGroupLayout = nullptr;
    wgpu::BindGroupLayout lineBindGroupLayout = nullptr;
    wgpu::BindGroupLayout gridBindGroupLayout = nullptr;

    // Vertex buffers
    wgpu::Buffer atomQuadVb = nullptr;
    wgpu::Buffer bondVb = nullptr;
    wgpu::Buffer boxVb = nullptr;
    wgpu::Buffer gridLineVb = nullptr;
    wgpu::Buffer gridInstVb = nullptr;

    // Storage buffers
    wgpu::Buffer sbPos = nullptr;    // array<vec4f> — x,y,z,pad
    wgpu::Buffer sbVel = nullptr;    // array<vec4f> — vx,vy,vz,pad
    wgpu::Buffer sbType = nullptr;   // array<f32>
    wgpu::Buffer sbRadius = nullptr; // array<f32>
    wgpu::Buffer sbSel = nullptr;    // array<f32>

    size_t sbCapacity_ = 0;
    size_t bondVbCapacity_ = 0;
    size_t gridInstVbCapacity_ = 0;

    wgpu::BindGroup atomBindGroup = nullptr;

    // Текущий render pass (живёт в течение drawShot)
    wgpu::RenderPassEncoder currentPass = nullptr;
    wgpu::TextureView currentTargetView = nullptr;
    wgpu::TextureView currentDepthView = nullptr;

    wgpu::TextureFormat surfaceFormat;

    void initAtomColors();
    void initAtomQuadBuffer();
    void initBoxBuffer();
    void initBondBuffer();
    void initGridLineBuffer();
    void initLinePipeline(wgpu::RenderPipeline& outPipeline, std::string_view wgsl);

    // Helpers
    wgpu::Buffer createBuffer(uint64_t size, wgpu::BufferUsage usage, std::string_view label);
    void ensureStorageBuffers(size_t count);
    template <typename T> void uploadStorageBuffer(wgpu::Buffer& buf, const T* data, size_t count);

    // Draw
    void drawAtomsImpl(const AtomStorage& atoms);
    void drawBondsImpl(const AtomStorage& atoms, const Bond::List& bonds);
    void drawBoxImpl(const World& box);
    void drawGridImpl(const SpatialGrid& grid);

    // Data
    struct GridInstance {
        glm::vec4 origin;
        float cellSize;
        float atomCount;
        float pad[2] = {};
    };

    std::vector<GridInstance> gridData;
    std::vector<glm::vec4> typeColorsData;
    std::vector<float> selectedData;
    std::vector<float> radii;
    std::vector<float> typeData;

    wgpu::CommandEncoder currentEncoder = nullptr;
};