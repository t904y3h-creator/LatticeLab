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
@group(0) @binding(0) var<uniform>       uScene  : SceneUniforms;
@group(0) @binding(1) var<storage, read> sPos    : array<vec4f>;
@group(0) @binding(2) var<storage, read> sVel    : array<vec4f>;
@group(0) @binding(3) var<storage, read> sType   : array<f32>;
@group(0) @binding(4) var<storage, read> sRadius : array<f32>;
@group(0) @binding(5) var<storage, read> sSel    : array<u32>;

struct VertOut {
    @builtin(position) pos    : vec4f,
    @location(0)       uv     : vec2f,
    @location(1)       color  : vec3f,
    @location(2)       sel    : u32,
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
    @location(0)        quadPos     : vec2f,
    @builtin(instance_index) iid    : u32,
) -> VertOut {
    let pos = sPos[iid].xyz;
    let r  = sRadius[iid];

    let right = vec3f(uScene.view[0][0], uScene.view[1][0], uScene.view[2][0]);
    let up    = vec3f(uScene.view[0][1], uScene.view[1][1], uScene.view[2][1]);

    let worldPos = pos
                 + right * quadPos.x * r
                 + up    * quadPos.y * r;

    var color: vec3f;
    if (uScene.colorMode == 0u) {
        let t = u32(sType[iid]);
        color = uScene.typeColors[t].rgb;
    }
    else {
        let v = sVel[iid];
        let speedSqr = v.x*v.x + v.y*v.y + v.z*v.z;
        let t = clamp(speedSqr / uScene.maxSpeedSqr, 0.0, 1.0);
        if (uScene.colorMode == 1u) {
            color = vec3f(t, 0.0, 1.0 - t);
        }
        else {
            color = turboColor(t);
        }
    }

    return VertOut(
        uScene.projection * uScene.view * vec4f(worldPos, 1.0);
        quadPos; 
        color;
        sSel[iid]
    );
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let d2 = dot(in.uv, in.uv);
    if (d2 > 1.0) { discard; }

    var color = in.color;

    let z = sqrt(1.0 - d2);
    let n = normalize(vec3f(in.uv, z));
    let light = normalize(uScene.lightDir.xyz);
    let diff  = max(dot(n, light), 0.0);
    color = vec3f(color.rgb * (0.3 + 0.7 * diff));

    if (in.sel != 0) {
        color = mix(color, vec3f(1.0, 0.85, 0.0), 0.5);
    }

    return vec4f(color, 1.0);
}
