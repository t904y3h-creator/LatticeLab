struct Uniforms {
    dt           : f32,
    accel_damping: f32,
    atom_count   : u32,
    _pad         : u32,
};
 
@group(0) @binding(0) var<uniform>             u         : Uniforms;
@group(0) @binding(1) var<storage, read_write> vel       : array<vec4f>; // vx,vy,vz,_
@group(0) @binding(2) var<storage, read>       force     : array<vec4f>; // fx,fy,fz,_
@group(0) @binding(3) var<storage, read>       prev_force: array<vec4f>; // pfx,pfy,pfz,_
@group(0) @binding(4) var<storage, read>       inv_mass  : array<f32>;
 
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= u.atom_count) { return; }
 
    let half_dt_inv_m = 0.5 * u.accel_damping * u.dt * inv_mass[i];
 
    vel[i] = vec4f(
        vel[i].xyz + (prev_force[i].xyz + force[i].xyz) * half_dt_inv_m,
        0.0
    );
}
