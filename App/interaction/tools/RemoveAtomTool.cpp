#include "RemoveAtomTool.h"

#include <vector>

#include "App/interaction/picking/PickingSystem.h"
#include "Engine/World.h"
#include "GUI/interface/UiState.h"

RemoveAtomTool::RemoveAtomTool(ToolContext& context) noexcept : ITool(context) {}

void RemoveAtomTool::onLeftPressed(Vec2i mousePos) {
    ToolContext& ctx = context();
    assert(ctx.world != nullptr && ctx.pickingSystem != nullptr && ctx.uiState != nullptr);
    if (ctx.world->atomCount() == 0) [[unlikely]] {
        return;
    }

    AtomHit hit;
    if (!ctx.pickingSystem->pickAtom(mousePos, 10.0f, hit)) {
        return;
    }

    const size_t target = hit.index;
    const auto& selected = ctx.pickingSystem->getSelectedIndices();

    if (selected.contains(target)) {
        std::vector<size_t> toRemove(selected.begin(), selected.end());
        for (size_t idx : toRemove) {
            ctx.pickingSystem->handleAtomRemoval(idx);
        }
        ctx.world->removeAtoms(toRemove);
    }
    else {
        ctx.pickingSystem->handleAtomRemoval(target);
        ctx.world->removeAtoms({&target, 1});
    }

    ctx.uiState->selectedAtomCount = static_cast<int>(ctx.pickingSystem->getSelectedIndices().size());
}
