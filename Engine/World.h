#pragma once

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/neigbors/GpuGridBuffers.h"
#include "Engine/math/Vec3.h"

class World {
public:
    World(const Vec3f& worldsize, size_t countAtoms, float gridCellSize = 6.f);

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    void setWorldSize(const Vec3f& newSize);

    void setGridCellSize(float newSize);

private:
    void rebuildGrid();

    Vec3f size;
    Vec3f gravity;

    GpuAtomBuffers atomBuffers;

    float gridCellSize;
    GpuGridBuffers gridBuffers;
};
