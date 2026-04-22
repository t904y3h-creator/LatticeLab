struct SceneUniforms {
    view       : mat4x4f,
    projection : mat4x4f,
    lightDir   : vec4f,
    colorMode  : vec4f,
    maxSpeedSqr: vec4f,
    maxCount   : vec4f,
    typeColors : array<vec4f, 119>,
}
@group(0) @binding(0) var<uniform> uScene: SceneUniforms;

struct VertOut {
    @builtin(position) pos   : vec4f,
    @location(0)       color : vec4f,
}

@vertex
fn vs_main(
    @location(0) localPos  : vec3f,
    @location(1) origin    : vec4f,
    @location(2) cellSize  : f32,
    @location(3) atomCount : f32,
) -> VertOut {
    let worldPos = origin.xyz + localPos * cellSize;
    let t = clamp(atomCount / uScene.maxCount.x, 0.0, 1.0);
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
