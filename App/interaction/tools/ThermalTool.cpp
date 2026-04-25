#include "ThermalTool.h"

#include <algorithm>
#include <cmath>

#include "Rendering/BaseRenderer.h"
#include "Rendering/camera/Camera.h"

namespace {
    constexpr float kMinRadius = 0.25f;
    constexpr float kMaxRadius = 100.0f;
    constexpr float kMinStrength = 0.0f;
    constexpr float kMaxStrength = 10.0f;
}

ThermalTool::ThermalTool(ToolContext& context) noexcept : ITool(context) {}

void ThermalTool::onLeftPressed(Vec2i mousePos) {
    active_ = true;
    applyAt(mousePos, 1.0f / 60.0f);
}

void ThermalTool::onLeftReleased(Vec2i mousePos) {
    (void)mousePos;
    active_ = false;
}

void ThermalTool::onFrame(Vec2i mousePos, float deltaTime) {
    if (!active_) {
        return;
    }
    applyAt(mousePos, deltaTime);
}

void ThermalTool::reset() { active_ = false; }

void ThermalTool::setRadius(float radius) noexcept { radius_ = std::clamp(radius, kMinRadius, kMaxRadius); }

void ThermalTool::setStrength(float strength) noexcept { strength_ = std::clamp(strength, kMinStrength, kMaxStrength); }

void ThermalTool::applyAt(Vec2i mousePos, float deltaTime) {
    ToolContext& ctx = context();
    if (!ctx.isValid() || strength_ <= 0.f) {
        return;
    }

    IRenderer* renderer = ctx.activeRenderer();
    if (renderer == nullptr) {
        return;
    }

    World& world = *ctx.world;
    const size_t mobileCount = world.mobileCount();
    if (mobileCount == 0) {
        return;
    }

    // TODO: заменить на compute шейдер
    std::vector<Vec3f> velocities(world.atomCount());
    world.getAtomBuffers().downloadVelocities(velocities);

    const Vec3f center = screenToWorld(mousePos);
    const float radiusSqr = radius_ * radius_;
    const bool is2D = renderer->camera.getMode() == Camera::Mode::Mode2D;
    const float sign = (mode_ == Mode::Heat) ? 1.f : -1.f;
    const float amount = strength_ * deltaTime;

    for (size_t i = 0; i < mobileCount; ++i) {
        const float dx = velocities[i].x - center.x;
        const float dy = velocities[i].y - center.y;
        const float dz = is2D ? 0.f : velocities[i].z - center.z;
        const float distSqr = dx * dx + dy * dy + dz * dz;
        if (distSqr > radiusSqr) {
            continue;
        }

        const float falloff = 1.f - std::sqrt(distSqr) / radius_;
        const float clampedFactor = std::max(0.f, 1.f + sign * amount * falloff);
        velocities[i] *= clampedFactor;
    }

    world.getAtomBuffers().uploadVelocities(velocities);
}
