#include "GpuLJForceField.h"

#include <array>
#include <cassert>
#include <vector>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/gpu/neigbors/GpuSpatialGrid.h"
#include "Engine/physics/AtomStorage.h"
#include "Engine/physics/ForceFields/LJForceField.h"
#include "generated/shaders/force_lj.wgsl.h"

struct LJUniforms {
    uint32_t atomCount;
    uint32_t typeCount;
    float cutoffSqr;

    uint32_t grid_dx;
    uint32_t grid_dy;
    uint32_t grid_dz;
    float grid_cell_size;

    uint32_t _pad;
};
static_assert(sizeof(LJUniforms) == 32);

void GpuLJForceField::init(const LJForceField& ljForceField, const AtomStorage& atoms) {
    typeCount_ = static_cast<uint32_t>(AtomData::Type::COUNT);

    buildPipeline();
    uploadLJTable(ljForceField);
    uploadAtomTypes(atoms);
}

void GpuLJForceField::buildPipeline() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(force_ljWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 8> bglEntries{};

    auto makeStorageBGLE = [](uint32_t binding, wgpu::BufferBindingType type) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        e.buffer.minBindingSize = 0;
        return e;
    };

    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Compute;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = sizeof(LJUniforms);

    bglEntries[1] = makeStorageBGLE(1, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[2] = makeStorageBGLE(2, wgpu::BufferBindingType::Storage);
    bglEntries[3] = makeStorageBGLE(3, wgpu::BufferBindingType::Storage);
    bglEntries[4] = makeStorageBGLE(4, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[5] = makeStorageBGLE(5, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[6] = makeStorageBGLE(6, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[7] = makeStorageBGLE(7, wgpu::BufferBindingType::ReadOnlyStorage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = WGPUContext::instance().device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pipelineLayout_;
    cpDesc.compute.module = shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = WGPUContext::instance().device().createComputePipeline(cpDesc);

    wgpu::BufferDescriptor ubDesc{};
    ubDesc.label = wgpu::StringView("LJ_Uniforms");
    ubDesc.size = sizeof(LJUniforms);
    ubDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    ubDesc.mappedAtCreation = false;
    uniformBuffer_ = WGPUContext::instance().device().createBuffer(ubDesc);
}

void GpuLJForceField::uploadLJTable(const LJForceField& ljForceField) {
    // Плоский массив [typeCount × typeCount] пар (C6, C12)
    // lj_table[i * typeCount + j] = {C6, C12} для пары типов (i, j)
    const size_t n = typeCount_;
    std::vector<float> table(n * n * 2);

    for (size_t i = 0; i < n; ++i) {
        const auto& row = ljForceField.pairRow(static_cast<AtomData::Type>(i));
        for (size_t j = 0; j < n; ++j) {
            table[(i * n + j) * 2 + 0] = row[j].potentialC6;
            table[(i * n + j) * 2 + 1] = row[j].potentialC12;
        }
    }

    const size_t bytes = table.size() * sizeof(float);
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView("LJ_Table");
    desc.size = bytes;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.mappedAtCreation = false;
    ljTableBuffer_ = WGPUContext::instance().device().createBuffer(desc);
    WGPUContext::instance().queue().writeBuffer(ljTableBuffer_, 0, table.data(), bytes);
}

void GpuLJForceField::uploadAtomTypes(const AtomStorage& atoms) {
    const size_t n = atoms.size();
    std::vector<uint32_t> types(n);
    for (size_t i = 0; i < n; ++i) {
        types[i] = static_cast<uint32_t>(atoms.type(i));
    }

    const size_t bytes = n * sizeof(uint32_t);
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView("AtomTypes");
    desc.size = bytes;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.mappedAtCreation = false;
    atomTypesBuffer_ = WGPUContext::instance().device().createBuffer(desc);
    WGPUContext::instance().queue().writeBuffer(atomTypesBuffer_, 0, types.data(), bytes);
}

wgpu::BindGroup GpuLJForceField::makeBindGroup(GpuAtomBuffers& atomBufs, GpuSpatialGrid& gridBufs) const {
    const size_t cap = atomBufs.countAtoms();
    const size_t vec4Bytes = cap * 4 * sizeof(float);
    const size_t f32Bytes = cap * sizeof(float);
    const size_t nlOffBytes = (nlBufs.atomCount() + 1) * sizeof(uint32_t);
    const size_t nlNbrBytes = nlBufs.fullPairCount() * sizeof(uint32_t);
    const size_t tableBytes = typeCount_ * typeCount_ * 2 * sizeof(float);
    const size_t typeBytes = cap * sizeof(uint32_t);

    std::array<wgpu::BindGroupEntry, 8> entries{};

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = size;
        return e;
    };

    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer_;
    entries[0].size = sizeof(LJUniforms);
    entries[1] = makeBE(1, atomBufs.bufPos(), vec4Bytes);
    entries[2] = makeBE(2, atomBufs.bufF(), vec4Bytes);
    entries[3] = makeBE(3, atomBufs.bufEnergy(), f32Bytes);
    entries[4] = makeBE(4, nlBufs.bufOffsets(), nlOffBytes);
    entries[5] = makeBE(5, nlBufs.bufNeighbors(), nlNbrBytes);
    entries[6] = makeBE(6, ljTableBuffer_, tableBytes);
    entries[7] = makeBE(7, atomTypesBuffer_, typeBytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();
    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuLJForceField::record(wgpu::CommandEncoder enc, GpuAtomBuffers& atomBufs, GpuSpatialGrid& gridBufs, uint32_t atomCount,
                             float cutoffSqr) {
    assert(isReady());

    LJUniforms uni{atomCount, typeCount_, cutoffSqr, gridBufs.0u};
    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bg = makeBindGroup(atomBufs, gridBufs);

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("LJ_forces pass");
    auto pass = enc.beginComputePass(pd);
    pass.setPipeline(pipeline_);
    pass.setBindGroup(0, bg, 0, nullptr);
    pass.dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass.end();
}