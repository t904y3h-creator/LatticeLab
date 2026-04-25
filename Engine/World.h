#pragma once

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/neigbors/GpuGridBuffers.h"
#include "Engine/physics/Bond.h"
#include "Engine/physics/ForceFields/LJTable.h"

class World {
public:
    World(const Vec3f& worldsize, size_t countAtoms, float gridCellSize = 6.f);

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    void clear();

    void setWorldSize(const Vec3f& newSize);
    const Vec3f& getWorldSize() const noexcept { return size; }

    void setGravity(const Vec3f& g) { gravity = g; }
    const Vec3f& getGravity() const noexcept { return gravity; }

    void setGridCellSize(float newSize);
    float getGridCellSize() const noexcept { return gridCellSize; }

    bool isLJEnabled() const { return ljEnabled; }
    void setLJEnabled(bool v) { ljEnabled = v; }
    bool isCoulombEnabled() const { return coulombEnabled; }
    void setCoulombEnabled(bool v) { coulombEnabled = v; }

    const Vec3u& getGridSize() const noexcept { return gridSize; }

    const GpuAtomBuffers& getAtomBuffers() const { return atomBuffers; }
    GpuAtomBuffers& getAtomBuffers() { return atomBuffers; }
    const GpuGridBuffers& getGridBuffers() const { return gridBuffers; }
    GpuGridBuffers& getGridBuffers() { return gridBuffers; }

    size_t atomCount() const { return atomBuffers.countAtoms(); }
    size_t mobileCount() const { return mobileCount_; }
    size_t staticCount() const { return atomCount() - mobileCount_; }

    const LJTable& getLJForceField() const { return ljForceField_; }

    // Добавляет подвижные атомы (вставляет перед статичными).
    void addAtoms(std::span<const Vec3f> positions, std::span<const Vec3f> velocities, std::span<const uint32_t> types);
    // Добавляет статичные атомы (в конец).
    void addStaticAtoms(std::span<const Vec3f> positions, std::span<const uint32_t> types);
    // Удаляет атомы по индексам.
    void removeAtoms(std::span<const size_t> index);
    // Полностью очищает все атомы.
    void clearAtoms();

    Bond::List& bonds() { return bonds_; }
    const Bond::List& bonds() const { return bonds_; }

private:
    void resizeGrid();
    void downloadAll(std::vector<Vec3f>& positions, std::vector<Vec3f>& velocities, std::vector<Vec3f>& forces,
                     std::vector<float>& invMasses, std::vector<float>& charges, std::vector<float>& pe, std::vector<uint32_t>& types,
                     std::vector<uint32_t>& valences) const;
    void uploadAll(const std::vector<Vec3f>& positions, const std::vector<Vec3f>& velocities, const std::vector<Vec3f>& forces,
                   const std::vector<float>& invMasses, const std::vector<float>& charges, const std::vector<float>& pe,
                   const std::vector<uint32_t>& types, const std::vector<uint32_t>& valences);

    bool ljEnabled = true;
    bool coulombEnabled = false;

    Vec3f size;
    Vec3f gravity;
    LJTable ljForceField_;

    size_t mobileCount_ = 0;
    GpuAtomBuffers atomBuffers;

    float gridCellSize;
    Vec3u gridSize;
    GpuGridBuffers gridBuffers;

    Bond::List bonds_;
};
