#include "World.h"

World::World(const Vec3f& worldsize, size_t countAtoms, float gridCellSize) : size(worldsize), gridCellSize(gridCellSize) {
    atomBuffers.resize(countAtoms);

    rebuildGrid();
}

void World::setWorldSize(const Vec3f& newSize) {
    size = newSize;

    rebuildGrid();
}

void World::setGridCellSize(float newSize) {
    gridCellSize = newSize;

    rebuildGrid();
}

void World::rebuildGrid() {
    Vec3u gridSize(size / gridCellSize);
    gridBuffers.resize(atomBuffers.countAtoms(), gridSize.x * gridSize.y * gridSize.z);
}
