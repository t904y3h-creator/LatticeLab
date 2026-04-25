#include "PickingSystem.h"

#include <limits>

#include "Engine/World.h"
#include "Engine/math/Ray.h"
#include "Engine/physics/AtomData.h"
#include "Rendering/BaseRenderer.h"

PickingSystem::PickingSystem(World& world, std::unique_ptr<IRenderer>& renderer) : world(world), renderer(&renderer) {}

void PickingSystem::clearSelection() {
    overlay.reset();
    selectedIndices.clear();
}

void PickingSystem::downloadAtomData() const {
    const size_t count = world.getAtomBuffers().countAtoms();
    cachedPositions_.resize(count);
    cachedTypes_.resize(count);
    world.getAtomBuffers().downloadPositions(cachedPositions_);
    world.getAtomBuffers().downloadAtomType(cachedTypes_);
}

void PickingSystem::processClick(Vec2i screenPos, bool cumulative) {
    AtomHit hit;
    bool found = pickAtom(screenPos, 10.0f, hit);

    if (found) {
        if (cumulative && selectedIndices.contains(hit.index)) {
            selectedIndices.erase(hit.index);
        }
        else {
            selectedIndices.insert(hit.index);
        }
    }
    else {
        if (!cumulative) {
            clearSelection();
        }
    }
}

void PickingSystem::processRect(Vec2i start, Vec2i end, bool cumulative) {
    if (!cumulative) {
        clearSelection();
    }

    downloadAtomData();
    IRenderer* rend = renderer->get();

    for (size_t i = 0; i < cachedPositions_.size(); ++i) {
        const Vec2i atomScreen = rend->camera.worldToScreen(cachedPositions_[i]);
        if (pointInRect(atomScreen, start, end)) {
            selectedIndices.insert(i);
        }
    }
}

void PickingSystem::processLasso(std::span<Vec2i> points, bool cumulative) {
    if (points.size() < 3) {
        return;
    }
    if (!cumulative) {
        clearSelection();
    }

    downloadAtomData();
    IRenderer* rend = renderer->get();

    for (size_t i = 0; i < cachedPositions_.size(); ++i) {
        const Vec2i screenPos = rend->camera.worldToScreen(cachedPositions_[i]);
        if (pointInPolygon(screenPos, points)) {
            selectedIndices.insert(i);
        }
    }
}

void PickingSystem::handleAtomRemoval(size_t index) {
    selectedIndices.erase(index);
    const size_t movedFrom = world.getAtomBuffers().countAtoms();
    if (index < movedFrom) {
        if (selectedIndices.erase(movedFrom) > 0) {
            selectedIndices.insert(index);
        }
    }
}

bool PickingSystem::pickAtom(Vec2i screenPos, float tolerance, AtomHit& hit) const {
    downloadAtomData();
    IRenderer* rend = renderer->get();
    switch (rend->camera.getMode()) {
    case Camera::Mode::Mode2D:
        return pickAtom2D(screenPos, tolerance, hit);
    case Camera::Mode::Orbit:
    case Camera::Mode::Free:
        return pickAtom3D(screenPos, hit);
    }
    return false;
}

bool PickingSystem::pickAtom2D(Vec2i screenPos, float tolerance, AtomHit& hit) const {
    IRenderer* rend = renderer->get();
    float bestDistSqr = std::numeric_limits<float>::max();
    size_t bestIndex = static_cast<size_t>(-1);

    for (size_t i = 0; i < cachedPositions_.size(); ++i) {
        const Vec2i atomScreen = rend->camera.worldToScreen(cachedPositions_[i]);
        const float distSqr = (atomScreen - screenPos).sqrAbs();
        const float atomRadius = AtomData::getProps(static_cast<AtomData::Type>(cachedTypes_[i])).radius;
        const float screenRadius = atomRadius * rend->camera.getZoom() + tolerance;

        if (distSqr < screenRadius * screenRadius && distSqr < bestDistSqr) {
            bestDistSqr = distSqr;
            bestIndex = i;
        }
    }

    if (bestIndex == static_cast<size_t>(-1)) {
        return false;
    }
    hit = {bestIndex, std::sqrt(bestDistSqr)};
    return true;
}

bool PickingSystem::pickAtom3D(Vec2i screenPos, AtomHit& hit) const {
    const Ray ray = (*renderer)->camera.screenToRay(static_cast<float>(screenPos.x), static_cast<float>(screenPos.y));

    float bestT = std::numeric_limits<float>::max();
    size_t bestIndex = static_cast<size_t>(-1);

    for (size_t i = 0; i < cachedPositions_.size(); ++i) {
        const float radius = AtomData::getProps(static_cast<AtomData::Type>(cachedTypes_[i])).radius;
        RaySphereHit rayHit;
        if (raySphereIntersect(ray, cachedPositions_[i], radius, rayHit)) {
            if (rayHit.t < bestT) {
                bestT = rayHit.t;
                bestIndex = i;
            }
        }
    }

    if (bestIndex == static_cast<size_t>(-1)) {
        return false;
    }
    hit = {bestIndex, bestT};
    return true;
}

// Ray casting алгоритм для point-in-polygon
template <typename T> bool PickingSystem::pointInPolygon(Vec2<T> point, std::span<Vec2<T>> polygon) {
    bool inside = false;
    const int x = point.x;
    const int y = point.y;
    const size_t n = polygon.size();

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const int xi = polygon[i].x, yi = polygon[i].y;
        const int xj = polygon[j].x, yj = polygon[j].y;

        const bool intersects = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersects) {
            inside = !inside;
        }
    }

    return inside;
}

template <typename T> bool PickingSystem::pointInRect(Vec2<T> point, Vec2<T> start, Vec2<T> end) {
    int minX = std::min(start.x, end.x);
    int maxX = std::max(start.x, end.x);
    int minY = std::min(start.y, end.y);
    int maxY = std::max(start.y, end.y);
    return (minX <= point.x && point.x <= maxX && minY <= point.y && point.y <= maxY);
}
