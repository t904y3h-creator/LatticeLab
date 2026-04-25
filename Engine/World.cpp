#include "World.h"

#include <algorithm>

World::World(const Vec3f& worldsize, size_t countAtoms, float gridCellSize) : size(worldsize), gridCellSize(gridCellSize) {
    atomBuffers.resize(countAtoms);
    resizeGrid();
}

void World::setWorldSize(const Vec3f& newSize) {
    size = newSize;
    resizeGrid();
}

void World::setGridCellSize(float newSize) {
    gridCellSize = newSize;
    resizeGrid();
}

void World::resizeGrid() {
    gridSize = Vec3u(size / gridCellSize);
    gridBuffers.resize(atomBuffers.countAtoms(), gridSize.x * gridSize.y * gridSize.z);
}

void World::downloadAll(std::vector<Vec3f>& positions, std::vector<Vec3f>& velocities, std::vector<Vec3f>& forces,
                        std::vector<float>& invMasses, std::vector<float>& charges, std::vector<float>& pe, std::vector<uint32_t>& types,
                        std::vector<uint32_t>& valences) const {
    const size_t count = atomBuffers.countAtoms();
    positions.resize(count);
    velocities.resize(count);
    forces.resize(count);
    invMasses.resize(count);
    charges.resize(count);
    pe.resize(count);
    types.resize(count);
    valences.resize(count);

    atomBuffers.downloadPositions(positions);
    atomBuffers.downloadVelocities(velocities);
    atomBuffers.downloadForces(forces);
    atomBuffers.downloadInvMass(invMasses);
    atomBuffers.downloadCharge(charges);
    atomBuffers.downloadPe(pe);
    atomBuffers.downloadAtomType(types);
    atomBuffers.downloadValence(valences);
}

void World::uploadAll(const std::vector<Vec3f>& positions, const std::vector<Vec3f>& velocities, const std::vector<Vec3f>& forces,
                      const std::vector<float>& invMasses, const std::vector<float>& charges, const std::vector<float>& pe,
                      const std::vector<uint32_t>& types, const std::vector<uint32_t>& valences) {
    atomBuffers.uploadPositions(positions);
    atomBuffers.uploadVelocities(velocities);
    atomBuffers.uploadForces(forces);
    atomBuffers.uploadInvMass(invMasses);
    atomBuffers.uploadCharge(charges);
    atomBuffers.uploadPe(pe);
    atomBuffers.uploadAtomType(types);
    atomBuffers.uploadValence(valences);
}

void World::addAtoms(std::span<const Vec3f> positions, std::span<const Vec3f> velocities, std::span<const uint32_t> types) {
    const size_t oldCount = atomBuffers.countAtoms();
    const size_t addCount = positions.size();
    const size_t newCount = oldCount + addCount;

    std::vector<Vec3f> pos, vel, forces;
    std::vector<float> invMasses, charges, pe;
    std::vector<uint32_t> typ, valences;
    downloadAll(pos, vel, forces, invMasses, charges, pe, typ, valences);

    // вставляем перед статичными
    pos.insert(pos.begin() + mobileCount_, positions.begin(), positions.end());
    vel.insert(vel.begin() + mobileCount_, velocities.begin(), velocities.end());
    forces.insert(forces.begin() + mobileCount_, addCount, Vec3f(0.f, 0.f, 0.f));
    invMasses.insert(invMasses.begin() + mobileCount_, addCount, 1.f);
    charges.insert(charges.begin() + mobileCount_, addCount, 0.f);
    pe.insert(pe.begin() + mobileCount_, addCount, 0.f);
    typ.insert(typ.begin() + mobileCount_, types.begin(), types.end());
    valences.insert(valences.begin() + mobileCount_, addCount, 0u);

    mobileCount_ += addCount;

    atomBuffers.resize(newCount);
    resizeGrid();
    uploadAll(pos, vel, forces, invMasses, charges, pe, typ, valences);
}

void World::addStaticAtoms(std::span<const Vec3f> positions, std::span<const uint32_t> types) {
    const size_t oldCount = atomBuffers.countAtoms();
    const size_t addCount = positions.size();
    const size_t newCount = oldCount + addCount;

    std::vector<Vec3f> pos, vel, forces;
    std::vector<float> invMasses, charges, pe;
    std::vector<uint32_t> typ, valences;
    downloadAll(pos, vel, forces, invMasses, charges, pe, typ, valences);

    // статичные просто в конец
    pos.insert(pos.end(), positions.begin(), positions.end());
    vel.insert(vel.end(), addCount, Vec3f(0.f, 0.f, 0.f));
    forces.insert(forces.end(), addCount, Vec3f(0.f, 0.f, 0.f));
    invMasses.insert(invMasses.end(), addCount, 0.f); // invMass = 0 => статичный
    charges.insert(charges.end(), addCount, 0.f);
    pe.insert(pe.end(), addCount, 0.f);
    typ.insert(typ.end(), types.begin(), types.end());
    valences.insert(valences.end(), addCount, 0u);

    atomBuffers.resize(newCount);
    resizeGrid();
    uploadAll(pos, vel, forces, invMasses, charges, pe, typ, valences);
}

void World::removeAtoms(std::span<const size_t> indices) {
    std::vector<Vec3f> pos, vel, forces;
    std::vector<float> invMasses, charges, pe;
    std::vector<uint32_t> typ, valences;
    downloadAll(pos, vel, forces, invMasses, charges, pe, typ, valences);

    std::vector<size_t> sorted(indices.begin(), indices.end());
    std::ranges::sort(sorted, std::greater<size_t>{});

    for (const size_t idx : sorted) {
        if (idx < mobileCount_) {
            mobileCount_--;
        }
        pos.erase(pos.begin() + idx);
        vel.erase(vel.begin() + idx);
        forces.erase(forces.begin() + idx);
        invMasses.erase(invMasses.begin() + idx);
        charges.erase(charges.begin() + idx);
        pe.erase(pe.begin() + idx);
        typ.erase(typ.begin() + idx);
        valences.erase(valences.begin() + idx);
    }

    atomBuffers.resize(pos.size());
    resizeGrid();
    uploadAll(pos, vel, forces, invMasses, charges, pe, typ, valences);
}

void World::clearAtoms() {
    mobileCount_ = 0;
    atomBuffers.resize(0);
    resizeGrid();
}
