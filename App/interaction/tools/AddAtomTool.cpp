#include "AddAtomTool.h"

#include "Engine/World.h"
#include "Engine/physics/AtomData.h"
#include "GUI/interface/UiState.h"
#include "GUI/interface/panels/periodic/PeriodicPanel.h"

AddAtomTool::AddAtomTool(ToolContext& context) noexcept : ITool(context) {}

void AddAtomTool::onLeftPressed(Vec2i mousePos) {
    ToolContext& ctx = context();
    if (!ctx.isValid() || ctx.uiState == nullptr) {
        return;
    }

    World& world = *ctx.world;
    const AtomData::Type atomType = static_cast<AtomData::Type>(PeriodicPanel::decodeAtom(ctx.uiState->selectedAtom));
    const Vec3f spawnPos = screenToWorld(mousePos);

    const Vec3f& ws = world.getWorldSize();
    if (!(1.f <= spawnPos.x && spawnPos.x <= ws.x - 1.f && 1.f <= spawnPos.y && spawnPos.y <= ws.y - 1.f && 1.f <= spawnPos.z &&
          spawnPos.z <= ws.z - 1.f)) {
        return;
    }

    const size_t count = world.atomCount();
    std::vector<Vec3f> positions(count);
    std::vector<uint32_t> types(count);
    world.getAtomBuffers().downloadPositions(positions);
    world.getAtomBuffers().downloadAtomType(types);

    const float atomRadius = AtomData::getProps(atomType).radius;
    for (size_t i = 0; i < count; ++i) {
        const float radius = AtomData::getProps(static_cast<AtomData::Type>(types[i])).radius;
        if ((positions[i] - spawnPos).sqrAbs() <= 4.f * (radius + atomRadius) * (radius + atomRadius)) {
            return;
        }
    }

    const Vec3f spawnVel = Vec3f::Random() * 5.f;
    const uint32_t type = static_cast<uint32_t>(atomType);
    world.addAtoms({&spawnPos, 1}, {&spawnVel, 1}, {&type, 1});
}
