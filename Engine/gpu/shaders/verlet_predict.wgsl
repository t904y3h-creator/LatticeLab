struct Uniforms {
    dt         : f32,
    atom_count : u32,
    _pad0      : u32,
    _pad1      : u32,
};

@group(0) @binding(0) var<uniform>             u        : Uniforms;
@group(0) @binding(1) var<storage, read_write> pos      : array<vec4f>; // x,y,z,_
@group(0) @binding(2) var<storage, read>       vel      : array<vec4f>; // vx,vy,vz,_
@group(0) @binding(3) var<storage, read>       force    : array<vec4f>; // fx,fy,fz,_
@group(0) @binding(4) var<storage, read>       inv_mass : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= u.atom_count) { return; }

    let dt    = u.dt;
    let inv_m = inv_mass[i];
    let p     = pos[i].xyz;
    let v     = vel[i].xyz;
    let f     = force[i].xyz;

    pos[i] = vec4f(p + (v + f * inv_m * 0.5 * dt) * dt, 0.0);
}
