// Мягкие стены (степенной потенциал) + постоянная сила (гравитация).

struct WallUniforms {
    max_x      : f32,
    max_y      : f32,
    max_z      : f32,
    gravity_x  : f32,
    gravity_y  : f32,
    gravity_z  : f32,
    atom_count : u32,
    _pad       : u32,
};

@group(0) @binding(0) var<uniform>             u    : WallUniforms;
@group(0) @binding(1) var<storage, read>       pos  : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> force: array<vec4f>;

fn apply_wall(coord: f32, max: f32) -> f32 {
    const k      = 500.0;
    const border = 2.0;

    let pen_low  = border - coord;
    let pen_high = coord - (max - border);

    let p6 = pen_low * pen_low * pen_low * pen_low * pen_low * pen_low;
    let f_low  = select(0.0, p6 * k, pen_low  > 0.0);

    let q6 = pen_high * pen_high * pen_high * pen_high * pen_high * pen_high;
    let f_high = select(0.0, q6 * k, pen_high > 0.0);

    return f_low - f_high;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.atom_count) { return; }

    let p = pos[gid.x].xyz;

    let fx = apply_wall(p.x, u.max_x) + u.gravity_x;
    let fy = apply_wall(p.y, u.max_y) + u.gravity_y;
    let fz = apply_wall(p.z, u.max_z) + u.gravity_z;

    force[gid.x] = vec4f(force[gid.x].xyz + vec3f(fx, fy, fz), 0.0);
}
