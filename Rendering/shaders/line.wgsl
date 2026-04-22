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
fn vs_main(@location(0) pos: vec3f) -> VertOut {
    return VertOut(
        uScene.projection * uScene.view * vec4f(pos, 1.0),
        vec4f(0.4, 0.6, 1.0, 0.3)
    );
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    return in.color;
}
