struct SceneUniforms {
    view        : mat4x4f,
    projection  : mat4x4f,
    lightDir    : vec4f,
    colorMode   : vec4f,
    maxSpeedSqr : vec4f,
    maxCount    : vec4f,
    typeColors  : array<vec4f, 119>,
}
@group(0) @binding(0) var<uniform>       uScene   : SceneUniforms;
@group(0) @binding(1) var<storage, read> sPos     : array<vec4f>;
@group(0) @binding(2) var<storage, read> sVel     : array<vec4f>;
@group(0) @binding(3) var<storage, read> sType    : array<f32>;
@group(0) @binding(4) var<storage, read> sRadius  : array<f32>;
@group(0) @binding(5) var<storage, read> sSel     : array<f32>;

struct VertOut {
    @builtin(position) pos    : vec4f,
    @location(0)       color  : vec3f,
    @location(1)       uv     : vec2f,
    @location(2)       sel    : f32,
}

fn turboColor(t: f32) -> vec3f {
    let x = clamp(0.1 + t * 0.75, 0.1, 0.85);

    let c0 = vec3f(0.135725, 0.091412, 0.106667);
    let c1 = vec3f(4.597373, 2.185608, 12.592549);
    let c2 = vec3f(-42.327686, 4.805216, -60.109686);
    let c3 = vec3f(130.588706, -14.019451, 109.074510);
    let c4 = vec3f(-150.566627, 4.210863, -88.506588);
    let c5 = vec3f(58.137451, 2.774745, 26.818275);
    var res = c5;
    res = res * x + c4;
    res = res * x + c3;
    res = res * x + c2;
    res = res * x + c1;
    res = res * x + c0;
    return res;
}

@vertex
fn vs_main(
    @location(0)             quadPos : vec2f,
    @builtin(instance_index) iid     : u32,
) -> VertOut {
    let pos    = sPos[iid].xy;
    let vel    = sVel[iid].xyz;
    let radius = sRadius[iid];
    let aType  = u32(sType[iid]);
    let sel    = sSel[iid];

    let mode = u32(uScene.colorMode.x);
    var color: vec3f;
    if (mode == 0u) {
        color = uScene.typeColors[aType].rgb;
    }
    else {
        let vSqr = dot(vel, vel);
        let t    = clamp(sqrt(vSqr / uScene.maxSpeedSqr.x), 0.0, 1.0);
        if (mode == 1u) {
            color = vec3f(t, 0.0, 1.0 - t);
        }
        else {
            color = turboColor(t);
        }
    }

    let worldPos = vec4f(pos + quadPos * radius, 0.0, 1.0);

    var out: VertOut;
    return VertOut(
        uScene.projection * uScene.view * worldPos,
        color,
        quadPos,
        sel
    );
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let d2 = dot(in.uv, in.uv);
    if (d2 > 1.0) { discard; }

    let d       = sqrt(d2);
    let outline = step(0.9, d);

    var outlineColor = vec3f(0.05, 0.05, 0.05);
    if (in.sel > 0.5) {
        outlineColor = vec3f(0.95, 0.72, 0.28);
    }

    let color = mix(in.color, outlineColor, outline);
    return vec4f(color, 1.0);
}
