#pragma once

#include "Engine/math/Vec3.h"
#include "Engine/physics/AtomStorage.h"
#include "Engine/physics/Bond.h"
#include "Engine/physics/ForceFields/BondForceField.h"
#include "Engine/physics/ForceFields/CoulombForceField.h"
#include "Engine/physics/ForceFields/LJForceField.h"

class NeighborList;
class World;

class ForceField {
    friend class GpuPhysicsPipeline;

public:
    ForceField();

    void compute(AtomStorage& atoms, Bond::List& bonds, World& box, NeighborList& neighborList, bool allowBondFormation, float dt) const;

    void setGravity(Vec3f gravity = Vec3f(0, 5, 0)) { static_force_ = gravity; }
    Vec3f getGravity() const { return static_force_; }
    void setLJEnabled(bool enabled) { enableLJ_ = enabled; }
    void setCoulombEnabled(bool enabled) { enableCoulomb_ = enabled; }
    [[nodiscard]] bool isLJEnabled() const { return enableLJ_; }
    [[nodiscard]] bool isCoulombEnabled() const { return enableCoulomb_; }

private:
    void computePairInteractions(AtomStorage& atoms, NeighborList& neighborList) const;

    Vec3f static_force_;
    LJForceField ljForceField_;
    BondForceField bondForceField_;
    CoulombForceField coulombForceField_;
    bool enableLJ_ = true;
    bool enableCoulomb_ = true;
};
