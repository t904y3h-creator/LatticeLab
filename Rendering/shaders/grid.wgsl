struct SceneUniforms {
    view       : mat4x4f,
    projection : mat4x4f,
    lightDir   : vec4f,
    colorMode  : u32,
    maxSpeedSqr: f32,
    maxCount   : u32,
    _pad       : f32,
    typeColors : array<vec4f, 119>,
}

struct GridUniforms {
    gridSize : vec3u,
    cellSize : f32,
}

@group(0) @binding(0) var<uniform>       uScene    : SceneUniforms;
@group(0) @binding(1) var<storage, read> cellCounts : array<u32>;
@group(0) @binding(2) var<uniform>       uGrid     : GridUniforms;

struct VertOut {
    @builtin(position) pos    : vec4f,
    @location(0)       color  : vec4f,
}

@vertex
fn vs_main(
    @location(0)             localPos : vec3f,
    @builtin(instance_index) instIdx  : u32,
) -> VertOut {
    let cnt = cellCounts[instIdx];

    if (cnt == 0u) {
        return VertOut(vec4f(0.0, 0.0, -2.0, 1.0), vec4f(0.0));
    }

    let cx = instIdx % uGrid.gridSize.x;
    let cy = (instIdx / uGrid.gridSize.x) % uGrid.gridSize.y;
    let cz = instIdx / (uGrid.gridSize.x * uGrid.gridSize.y);

    let origin   = vec3f(f32(cx), f32(cy), f32(cz)) * uGrid.cellSize;
    let worldPos = origin + localPos * uGrid.cellSize;

    let t     = clamp(f32(cnt) / f32(uScene.maxCount), 0.0, 1.0);
    let color = mix(vec4f(0.0, 1.0, 0.0, 0.3), vec4f(1.0, 0.0, 0.0, 0.3), t);

    return VertOut(
        uScene.projection * uScene.view * vec4f(worldPos, 1.0),
        color
    );
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    return in.color;
}
