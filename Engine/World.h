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
    const Vec3f& getWorldSize() const noexcept { return size; }

    void setGridCellSize(float newSize);
    float getGridCellSize() const noexcept { return gridCellSize; }

    const Vec3u& getGridSize() const noexcept { return gridSize; }

    const GpuAtomBuffers& getAtomBuffers() const { return atomBuffers; }
    GpuAtomBuffers& getAtomBuffers() { return atomBuffers; }
    const GpuGridBuffers& getGridBuffers() const { return gridBuffers; }
    GpuGridBuffers& getGridBuffers() { return gridBuffers; }

private:
    void rebuildGrid();

    Vec3f size;
    Vec3f gravity;

    GpuAtomBuffers atomBuffers;

    float gridCellSize;
    Vec3u gridSize;
    GpuGridBuffers gridBuffers;
};
