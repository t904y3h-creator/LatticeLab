#include "RendererWGPU.h"

#include <cstring>

#include <webgpu/webgpu.hpp>

#include "App/interaction/ToolsManager.h"
#include "Engine/World.h"
#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"

#ifdef True
#undef True
#endif

#ifdef False
#undef False
#endif

namespace {
    wgpu::ShaderModule createShaderModule(std::string_view wgsl, std::string_view label) {
        WGPUShaderSourceWGSL wgslDesc{};
        wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslDesc.code = wgpu::StringView(wgsl);

        wgpu::ShaderModuleDescriptor desc{};
        desc.label = wgpu::StringView(label);
        desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgslDesc);
        return WGPUContext::instance().device().createShaderModule(desc);
    }
}

RendererWGPU::RendererWGPU(World& world, wgpu::TextureFormat surfaceFormat) : IRenderer(world), surfaceFormat(surfaceFormat) {
    uniformBuffer = WGPUContext::instance().createBuffer(sizeof(SceneUniforms), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                                         "RenderUniformBuffer");

    initAtomColors();
    initAtomQuadBuffer();
    initBoxBuffer();
    initBondBuffer();
    initGridLineBuffer();
}

void RendererWGPU::initAtomColors() {
    const int typeCount = static_cast<int>(AtomData::Type::COUNT);
    typeColorsData.resize(typeCount);
    for (int i = 0; i < typeCount; ++i) {
        const auto& props = AtomData::getProps(static_cast<AtomData::Type>(i));
        typeColorsData[i] = glm::vec4(props.color.r / 255.f, props.color.g / 255.f, props.color.b / 255.f, props.color.a / 255.f);
    }
}

void RendererWGPU::initAtomQuadBuffer() {
    constexpr float quad[] = {
        -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, 1.f,
    };
    atomQuadVb =
        WGPUContext::instance().createBuffer(sizeof(quad), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, "AtomQuadGeometry");
    WGPUContext::instance().queue().writeBuffer(atomQuadVb, 0, quad, sizeof(quad));
}

void RendererWGPU::initBoxBuffer() {
    boxVb =
        WGPUContext::instance().createBuffer(24 * 3 * sizeof(float), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, "BoxGeometry");
}

void RendererWGPU::initBondBuffer() {
    bondVb = WGPUContext::instance().createBuffer(128, wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, "BondGeometry");
    bondVbCapacity_ = 128;
}

void RendererWGPU::initGridLineBuffer() {
    const float lines[] = {
        0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1,
    };
    gridLineVb =
        WGPUContext::instance().createBuffer(sizeof(lines), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, "GridCellUnitLines");
    WGPUContext::instance().queue().writeBuffer(gridLineVb, 0, lines, sizeof(lines));
}

void RendererWGPU::initAtomPipeline(std::string_view atomWGSL) {
    wgpu::ShaderModule shader = createShaderModule(atomWGSL, "AtomShader");

    std::array<wgpu::BindGroupLayoutEntry, 6> entries;
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;

    for (int i = 1; i <= 5; ++i) {
        entries[i].binding = i;
        entries[i].visibility = wgpu::ShaderStage::Vertex;
        entries[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("AtomBindGroupLayout");
    bglDesc.entryCount = entries.size();
    bglDesc.entries = entries.data();
    atomBindGroupLayout = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("AtomPipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&atomBindGroupLayout;
    auto pipelineLayout = WGPUContext::instance().device().createPipelineLayout(plDesc);

    wgpu::VertexAttribute quadAttr;
    quadAttr.format = wgpu::VertexFormat::Float32x2;
    quadAttr.offset = 0;
    quadAttr.shaderLocation = 0;

    wgpu::VertexBufferLayout quadLayout;
    quadLayout.arrayStride = 2 * sizeof(float);
    quadLayout.stepMode = wgpu::VertexStepMode::Vertex;
    quadLayout.attributeCount = 1;
    quadLayout.attributes = &quadAttr;

    wgpu::ColorTargetState colorTarget;
    colorTarget.format = surfaceFormat;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::BlendState blend;
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.color.operation = wgpu::BlendOperation::Add;
    blend.alpha = blend.color;
    colorTarget.blend = &blend;

    wgpu::FragmentState fragState;
    fragState.module = shader;
    fragState.entryPoint = wgpu::StringView("fs_main");
    fragState.targetCount = 1;
    fragState.targets = &colorTarget;

    wgpu::DepthStencilState depthState;
    depthState.format = wgpu::TextureFormat::Depth24Plus;
    depthState.depthWriteEnabled = wgpu::OptionalBool::True;
    depthState.depthCompare = wgpu::CompareFunction::Less;

    wgpu::RenderPipelineDescriptor pDesc{};
    pDesc.label = wgpu::StringView("AtomPipeline");
    pDesc.layout = pipelineLayout;
    pDesc.vertex.module = shader;
    pDesc.vertex.entryPoint = wgpu::StringView("vs_main");
    pDesc.vertex.bufferCount = 1;
    pDesc.vertex.buffers = &quadLayout;
    pDesc.fragment = &fragState;
    pDesc.depthStencil = &depthState;
    pDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pDesc.multisample.count = 1;
    pDesc.multisample.mask = 0xFFFFFFFF;
    pDesc.multisample.alphaToCoverageEnabled = false;

    atomPipeline = WGPUContext::instance().device().createRenderPipeline(pDesc);
}

void RendererWGPU::initLinePipeline(wgpu::RenderPipeline& outPipeline, std::string_view wgsl) {
    wgpu::ShaderModule shader = createShaderModule(wgsl, "LineShader");

    wgpu::BindGroupLayoutEntry uboEntry{};
    uboEntry.binding = 0;
    uboEntry.visibility = wgpu::ShaderStage::Vertex;
    uboEntry.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("LineBindGroupLayout");
    bglDesc.entryCount = 1;
    bglDesc.entries = &uboEntry;
    auto bgl = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("LinePipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&bgl;

    wgpu::VertexAttribute attr{};
    attr.format = wgpu::VertexFormat::Float32x3;
    attr.offset = 0;
    attr.shaderLocation = 0;

    wgpu::VertexBufferLayout vbl{};
    vbl.arrayStride = 3 * sizeof(float);
    vbl.stepMode = wgpu::VertexStepMode::Vertex;
    vbl.attributeCount = 1;
    vbl.attributes = &attr;

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = surfaceFormat;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fragState{};
    fragState.module = shader;
    fragState.entryPoint = wgpu::StringView("fs_main");
    fragState.targetCount = 1;
    fragState.targets = &colorTarget;

    wgpu::RenderPipelineDescriptor pDesc{};
    pDesc.label = wgpu::StringView("LineRenderPipeline");
    pDesc.layout = WGPUContext::instance().device().createPipelineLayout(plDesc);
    pDesc.vertex.module = shader;
    pDesc.vertex.entryPoint = wgpu::StringView("vs_main");
    pDesc.vertex.bufferCount = 1;
    pDesc.vertex.buffers = &vbl;
    pDesc.fragment = &fragState;
    pDesc.primitive.topology = wgpu::PrimitiveTopology::LineList;
    pDesc.multisample.count = 1;
    pDesc.multisample.mask = 0xFFFFFFFF;
    pDesc.multisample.alphaToCoverageEnabled = false;

    wgpu::DepthStencilState depthState{};
    depthState.format = wgpu::TextureFormat::Depth24Plus;
    depthState.depthWriteEnabled = wgpu::OptionalBool::False;
    depthState.depthCompare = wgpu::CompareFunction::Less;
    pDesc.depthStencil = &depthState;

    outPipeline = WGPUContext::instance().device().createRenderPipeline(pDesc);

    lineBindGroupLayout = bgl;
}

void RendererWGPU::initGridPipeline(std::string_view gridWGSL) {
    wgpu::ShaderModule shader = createShaderModule(gridWGSL, "GridShader");

    // binding 0 — uniforms, binding 1 — cellCounts (storage)
    std::array<wgpu::BindGroupLayoutEntry, 3> bglEntries{};
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = sizeof(SceneUniforms);

    bglEntries[1].binding = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    bglEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bglEntries[1].buffer.minBindingSize = 0;

    bglEntries[2].binding = 2;
    bglEntries[2].visibility = wgpu::ShaderStage::Vertex;
    bglEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[2].buffer.minBindingSize = sizeof(GridUniforms);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("GridBindGroupLayout");
    bglDesc.entryCount = bglEntries.size();
    bglDesc.entries = bglEntries.data();
    gridBindGroupLayout = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = gridBindGroupLayout;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("GridPipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;

    wgpu::VertexAttribute vertAttr{};
    vertAttr.format = wgpu::VertexFormat::Float32x3;
    vertAttr.offset = 0;
    vertAttr.shaderLocation = 0;

    wgpu::VertexBufferLayout vertLayout{};
    vertLayout.arrayStride = 3 * sizeof(float);
    vertLayout.stepMode = wgpu::VertexStepMode::Vertex;
    vertLayout.attributeCount = 1;
    vertLayout.attributes = &vertAttr;

    wgpu::BlendState blend{};
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.color.operation = wgpu::BlendOperation::Add;
    blend.alpha = blend.color;

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = surfaceFormat;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;
    colorTarget.blend = &blend;

    wgpu::FragmentState fragState{};
    fragState.module = shader;
    fragState.entryPoint = wgpu::StringView("fs_main");
    fragState.targetCount = 1;
    fragState.targets = &colorTarget;

    wgpu::DepthStencilState depthState{};
    depthState.format = wgpu::TextureFormat::Depth24Plus;
    depthState.depthWriteEnabled = wgpu::OptionalBool::False;
    depthState.depthCompare = wgpu::CompareFunction::Less;

    wgpu::RenderPipelineDescriptor pDesc{};
    pDesc.label = wgpu::StringView("GridRenderPipeline");
    pDesc.layout = WGPUContext::instance().device().createPipelineLayout(plDesc);
    pDesc.vertex.module = shader;
    pDesc.vertex.entryPoint = wgpu::StringView("vs_main");
    pDesc.vertex.bufferCount = 1;
    pDesc.vertex.buffers = &vertLayout;
    pDesc.fragment = &fragState;
    pDesc.depthStencil = &depthState;
    pDesc.primitive.topology = wgpu::PrimitiveTopology::LineList;
    pDesc.multisample.count = 1;
    pDesc.multisample.mask = 0xFFFFFFFF;
    pDesc.multisample.alphaToCoverageEnabled = false;

    gridPipeline = WGPUContext::instance().device().createRenderPipeline(pDesc);

    gridUniformBuffer =
        WGPUContext::instance().createBuffer(sizeof(GridUniforms), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "GridUniforms");
}

void RendererWGPU::initBoxPipeline(std::string_view boxWGSL) { initLinePipeline(boxPipeline, boxWGSL); }
void RendererWGPU::initBondPipeline(std::string_view bondWGSL) { initLinePipeline(bondPipeline, bondWGSL); }

void RendererWGPU::rebuildBindGroup(const World& world) {
    const GpuAtomBuffers& atoms = world.getAtomBuffers();
    const size_t count = atoms.countAtoms();

    const uint64_t vec4Bytes = count * 4 * sizeof(float);
    const uint64_t f32Bytes = count * sizeof(float);
    const uint64_t u32Bytes = count * sizeof(uint32_t);
    const auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    sbRadius = WGPUContext::instance().createBuffer(f32Bytes, usage, "AtomsRadius");
    sbSel = WGPUContext::instance().createBuffer(u32Bytes, usage, "AtomsSel");

    std::array<wgpu::BindGroupEntry, 6> entries{};
    auto setStorageEntry = [](wgpu::BindGroupEntry& e, int binding, wgpu::Buffer buf, size_t size) {
        e.binding = binding;
        e.buffer = buf;
        e.size = size;
    };
    setStorageEntry(entries[0], 0, uniformBuffer, sizeof(SceneUniforms));
    setStorageEntry(entries[1], 1, atoms.bufPos(), vec4Bytes);
    setStorageEntry(entries[2], 2, atoms.bufVel(), vec4Bytes);
    setStorageEntry(entries[3], 3, atoms.bufAtomType(), u32Bytes);
    setStorageEntry(entries[4], 4, sbRadius, f32Bytes);
    setStorageEntry(entries[5], 5, sbSel, u32Bytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("AtomBindGroup");
    bgDesc.layout = atomBindGroupLayout;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    atomBindGroup = WGPUContext::instance().device().createBindGroup(bgDesc);
}

void RendererWGPU::drawShot(wgpu::CommandEncoder encoder, wgpu::TextureView targetView, wgpu::TextureView depthView, const World& world) {
    updateMatrices();

    SceneUniforms uniforms{};
    uniforms.view = view;
    uniforms.projection = projection;
    uniforms.lightDir = glm::vec4(getLightDir(), 0.f);
    uniforms.colorMode = glm::vec4(static_cast<float>(speedColorMode), 0, 0, 0);
    uniforms.maxSpeedSqr = glm::vec4(1.f, 0, 0, 0);
    uniforms.maxCount = glm::vec4(1.f, 0, 0, 0);
    for (size_t i = 0; i < typeColorsData.size(); ++i) {
        uniforms.typeColors[i] = typeColorsData[i];
    }

    WGPUContext::instance().device().getQueue().writeBuffer(uniformBuffer, 0, &uniforms, sizeof(uniforms));

    wgpu::RenderPassColorAttachment colorAtt;
    colorAtt.view = targetView;
    colorAtt.loadOp = wgpu::LoadOp::Clear;
    colorAtt.storeOp = wgpu::StoreOp::Store;
    colorAtt.clearValue = {33.0 / 255.0, 33.0 / 255.0, 33.0 / 255.0, 1.0};

    wgpu::RenderPassDepthStencilAttachment depthAtt;
    depthAtt.view = depthView;
    depthAtt.depthLoadOp = wgpu::LoadOp::Clear;
    depthAtt.depthStoreOp = wgpu::StoreOp::Store;
    depthAtt.depthClearValue = 1.0f;
    depthAtt.stencilLoadOp = wgpu::LoadOp::Undefined;
    depthAtt.stencilStoreOp = wgpu::StoreOp::Undefined;

    wgpu::RenderPassDescriptor passDesc{};
    passDesc.label = wgpu::StringView("RendererWGPU::drawShot pass");
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &colorAtt;
    passDesc.depthStencilAttachment = &depthAtt;

    currentPass = encoder.beginRenderPass(passDesc);

    if (drawBonds) {
        std::cerr << "Bonds not implemented yet" << std::endl;
        // drawBondsImpl(atoms, bonds);
    }
    if (drawGrid) {
        drawGridImpl(world);
    }
    drawBoxImpl(world.getWorldSize());
    drawAtomsImpl(world);
}

void RendererWGPU::drawAtomsImpl(const World& world) {
    const size_t count = world.getAtomBuffers().countAtoms();
    if (count == 0) {
        return;
    }

    // пересоздаём bind group если capacity изменился
    if (count != sbCapacity_) {
        rebuildBindGroup(world);
        uploadRadii(world);
        sbCapacity_ = count;
    }

    selectedData_.assign(count, 0u);
    for (const size_t idx : ToolsManager::pickingSystem->getSelectedIndices()) {
        if (idx < count) {
            selectedData_[idx] = 1u;
        }
    }
    WGPUContext::instance().queue().writeBuffer(sbSel, 0, selectedData_.data(), count * sizeof(uint32_t));

    // maxSpeedSqr для color mode
    float maxSpeedSqr = 1.f;
    if (speedColorMode != SpeedColorMode::AtomColor) {
        maxSpeedSqr = speedGradientMax > 0.f ? speedGradientMax * speedGradientMax
                                             : 1e-6f; // без CPU доступа к velocities больше не считаем per-frame max
    }
    WGPUContext::instance().queue().writeBuffer(uniformBuffer, offsetof(SceneUniforms, maxSpeedSqr), &maxSpeedSqr, sizeof(float));

    currentPass.setPipeline(atomPipeline);
    currentPass.setBindGroup(0, atomBindGroup, 0, nullptr);
    currentPass.setVertexBuffer(0, atomQuadVb, 0, atomQuadVb.getSize());
    currentPass.draw(6, static_cast<uint32_t>(count), 0, 0);
}

void RendererWGPU::drawBoxImpl(const Vec3f& worldSize) {
    const float x1 = worldSize.x, y1 = worldSize.y, z1 = worldSize.z;
    const float lines[] = {
        0, y1, 0, x1, y1, 0, x1, y1, 0, x1, y1, z1, x1, y1, z1, 0,  y1, z1, 0, y1, z1, 0, y1, 0,
        0, 0,  0, x1, 0,  0, x1, 0,  0, x1, 0,  z1, x1, 0,  z1, 0,  0,  z1, 0, 0,  z1, 0, 0,  0,
        0, 0,  0, 0,  y1, 0, x1, 0,  0, x1, y1, 0,  x1, 0,  z1, x1, y1, z1, 0, 0,  z1, 0, y1, z1,
    };
    WGPUContext::instance().device().getQueue().writeBuffer(boxVb, 0, lines, sizeof(lines));

    wgpu::BindGroupEntry entry;
    entry.binding = 0;
    entry.buffer = uniformBuffer;
    entry.size = sizeof(SceneUniforms);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("BoxBindGroup");
    bgDesc.layout = lineBindGroupLayout;
    bgDesc.entryCount = 1;
    bgDesc.entries = &entry;
    wgpu::BindGroup bg = WGPUContext::instance().device().createBindGroup(bgDesc);

    currentPass.setPipeline(boxPipeline);
    currentPass.setBindGroup(0, bg, 0, nullptr);
    currentPass.setVertexBuffer(0, boxVb, 0, sizeof(lines));
    currentPass.draw(24, 1, 0, 0);
}

// void RendererWGPU::drawBondsImpl(const AtomStorage& atoms, const Bond::List& bonds) {
//     if (bonds.empty()) {
//         return;
//     }

//     std::vector<glm::vec3> verts;
//     verts.reserve(bonds.size() * 2);
//     for (const Bond& bond : bonds) {
//         if (bond.aIndex >= atoms.size() || bond.bIndex >= atoms.size()) {
//             continue;
//         }
//         const Vec3f a = atoms.pos(bond.aIndex);
//         const Vec3f b = atoms.pos(bond.bIndex);
//         verts.emplace_back(a.x, a.y, a.z);
//         verts.emplace_back(b.x, b.y, b.z);
//     }
//     if (verts.empty()) {
//         return;
//     }

//     const uint64_t bytes = verts.size() * sizeof(glm::vec3);
//     if (bytes > bondVbCapacity_) {
//         bondVb = WGPUContext::instance().createBuffer(bytes * 2, wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, "BondGeometry");
//         bondVbCapacity_ = bytes * 2;
//     }
//     WGPUContext::instance().device().getQueue().writeBuffer(bondVb, 0, verts.data(), bytes);

//     wgpu::BindGroupEntry entry;
//     entry.binding = 0;
//     entry.buffer = uniformBuffer;
//     entry.size = sizeof(SceneUniforms);

//     wgpu::BindGroupDescriptor bgDesc{};
//     bgDesc.label = wgpu::StringView("BondBindGroup");
//     bgDesc.layout = lineBindGroupLayout;
//     bgDesc.entryCount = 1;
//     bgDesc.entries = &entry;
//     auto bg = WGPUContext::instance().device().createBindGroup(bgDesc);

//     currentPass.setPipeline(bondPipeline);
//     currentPass.setBindGroup(0, bg, 0, nullptr);
//     currentPass.setVertexBuffer(0, bondVb, 0, bytes);
//     currentPass.draw(verts.size(), 1, 0, 0);
// }

void RendererWGPU::drawGridImpl(const World& world) {
    const Vec3u& gs = world.getGridSize();
    const uint32_t totalCells = gs.x * gs.y * gs.z;
    assert(totalCells != 0);

    // TODO maxCount пока фиксированный, потом нужно шейдером вычислить
    float mc = 1.f;
    WGPUContext::instance().queue().writeBuffer(uniformBuffer, offsetof(SceneUniforms, maxCount), &mc, sizeof(float));

    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].size = sizeof(SceneUniforms);

    entries[1].binding = 1;
    entries[1].buffer = world.getGridBuffers().bufCount();
    entries[1].size = totalCells * sizeof(uint32_t);

    GridUniforms gridUni{};
    gridUni.cellSize = world.getGridCellSize();
    gridUni.dx = gs.x;
    gridUni.dy = gs.y;
    gridUni.dz = gs.z;
    WGPUContext::instance().queue().writeBuffer(gridUniformBuffer, 0, &gridUni, sizeof(gridUni));
    entries[2].binding = 2;
    entries[2].buffer = gridUniformBuffer;
    entries[2].size = sizeof(GridUniforms);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("GridBindGroup");
    bgDesc.layout = gridBindGroupLayout;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    auto bg = WGPUContext::instance().device().createBindGroup(bgDesc);

    currentPass.setPipeline(gridPipeline);
    currentPass.setBindGroup(0, bg, 0, nullptr);
    currentPass.setVertexBuffer(0, gridLineVb, 0, gridLineVb.getSize());
    currentPass.draw(24, totalCells, 0, 0);
}

void RendererWGPU::uploadRadii(const World& world) {
    const size_t count = world.getAtomBuffers().countAtoms();

    std::vector<uint32_t> types(count);
    world.getAtomBuffers().downloadAtomType(types);

    std::vector<float> radii(count);
    for (size_t i = 0; i < count; ++i) {
        radii[i] = AtomData::getProps(static_cast<AtomData::Type>(types[i])).radius;
    }
    WGPUContext::instance().queue().writeBuffer(sbRadius, 0, radii.data(), count * sizeof(float));
}
