#pragma once

#include <cstdint>

#include "Engine/World.h"
#include "Engine/physics/AtomData.h"

namespace Scenes {
    void crystal(World& world, int n, AtomData::Type type, bool is3d, double padding = 3.0, double margin = 15.0);

    int randomGasInCurrentBox(World& world, int atomCount, AtomData::Type type, bool is3d, float minDistance = 4.0f,
                              float speedScale = 5.0f, int maxAttemptsPerAtom = 20, uint32_t seed = 0);

    void randomGas(World& world, int atomCount, AtomData::Type type, bool is3d, double spacing = 6.0, double margin = 6.0,
                   float density = 1.0f, float speedScale = 5.0f, uint32_t seed = 0);
}
