// confine_to_box   - отражение от стенок (после predict)
// post_process_vel - ограничение максимальной скорости (после correct)

// --- confine_to_box ---
struct ConfineParams {
    max_x      : f32,
    max_y      : f32,
    max_z      : f32,
    restitution: f32,
    atom_count : u32,
    _pad0      : u32,
    _pad1      : u32,
    _pad2      : u32,
};

@group(0) @binding(0) var<uniform>             confine_u  : ConfineParams;
@group(0) @binding(1) var<storage, read_write> confine_pos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> confine_vel: array<vec4f>;

fn confine_axis(coord: f32, speed: f32, axis_max: f32, restitution: f32) -> vec2f {
    // возвращает vec2(новый coord, новый speed)
    if (coord < 0.0) {
        return vec2f(0.0, select(speed, -speed * restitution, speed < 0.0));
    }
    if (coord > axis_max) {
        return vec2f(axis_max, select(speed, -speed * restitution, speed > 0.0));
    }
    return vec2f(coord, speed);
}

@compute @workgroup_size(64)
fn confine_to_box(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= confine_u.atom_count) { return; }

    let p = confine_pos[gid.x].xyz;
    let v = confine_vel[gid.x].xyz;
    let r = confine_u.restitution;

    let rx = confine_axis(p.x, v.x, confine_u.max_x, r);
    let ry = confine_axis(p.y, v.y, confine_u.max_y, r);
    let rz = confine_axis(p.z, v.z, confine_u.max_z, r);

    confine_pos[gid.x] = vec4f(rx.x, ry.x, rz.x, 0.0);
    confine_vel[gid.x] = vec4f(rx.y, ry.y, rz.y, 0.0);
}

// --- post_process_vel ---
struct VelCapParams {
    max_speed_sqr: f32,
    max_speed    : f32,
    atom_count   : u32,
    _pad         : u32,
};

@group(0) @binding(0) var<uniform>             velcap_u  : VelCapParams;
@group(0) @binding(1) var<storage, read_write> velcap_vel: array<vec4f>;

@compute @workgroup_size(64)
fn post_process_vel(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= velcap_u.atom_count) { return; }

    let v         = velcap_vel[gid.x].xyz;
    let speed_sqr = dot(v, v);

    if (speed_sqr > velcap_u.max_speed_sqr) {
        let scale = velcap_u.max_speed / sqrt(speed_sqr);
        velcap_vel[gid.x] = vec4f(v * scale, 0.0);
    }
}