struct LJUniforms {
    atom_count : u32,
    type_count : u32,
    grid_dim_x : u32,
    grid_dim_y : u32,
    grid_dim_z : u32,
    cell_size  : f32,
    _pad       : u32,
    _pad2      : u32,
};

@group(0) @binding(0) var<uniform>             u              : LJUniforms;
@group(0) @binding(1) var<storage, read>       pos            : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> force          : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> energy         : array<f32>;
@group(0) @binding(4) var<storage, read>       grid_offsets   : array<u32>;
@group(0) @binding(5) var<storage, read>       sorted_indices : array<u32>;
@group(0) @binding(6) var<storage, read>       lj_table       : array<vec2f>;
@group(0) @binding(7) var<storage, read>       atom_types     : array<u32>;

fn cell_linear(p: vec3f) -> u32 {
    let cx = clamp(u32(p.x / u.cell_size), 0u, u.grid_dim_x - 1u);
    let cy = clamp(u32(p.y / u.cell_size), 0u, u.grid_dim_y - 1u);
    let cz = clamp(u32(p.z / u.cell_size), 0u, u.grid_dim_z - 1u);
    return cx + cy * u.grid_dim_x + cz * u.grid_dim_x * u.grid_dim_y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if i >= u.atom_count { return; }

    let pi      = pos[i].xyz;
    let type_i  = atom_types[i];
    let row_off = type_i * u.type_count;

    var fx = 0.0;
    var fy = 0.0;
    var fz = 0.0;
    var pe = 0.0;

    let c    = cell_linear(pi);
    let cx   = i32(c % u.grid_dim_x);
    let cy   = i32((c / u.grid_dim_x) % u.grid_dim_y);
    let cz   = i32(c / (u.grid_dim_x * u.grid_dim_y));

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nx = cx + dx;
                let ny = cy + dy;
                let nz = cz + dz;
                if (nx < 0 || ny < 0 || nz < 0 ||
                    nx >= i32(u.grid_dim_x) ||
                    ny >= i32(u.grid_dim_y) ||
                    nz >= i32(u.grid_dim_z)) { continue; }

                let ncell = u32(nx) + u32(ny) * u.grid_dim_x + u32(nz) * u.grid_dim_x * u.grid_dim_y;
                let begin = grid_offsets[ncell];
                let end   = grid_offsets[ncell + 1u];

                for (var p = begin; p < end; p++) {
                    let j = sorted_indices[p];
                    if i == j { continue; }

                    let pj = pos[j].xyz;
                    let dx = pj.x - pi.x;
                    let dy = pj.y - pi.y;
                    let dz = pj.z - pi.z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= 1e-6 { continue; }

                    let params  = lj_table[row_off + atom_types[j]];
                    let c6      = params.x;
                    let c12     = params.y;

                    let inv_d2  = 1.0 / d2;
                    let inv_d6  = inv_d2 * inv_d2 * inv_d2;
                    let inv_d12 = inv_d6 * inv_d6;

                    let term6   = c6  * inv_d6;
                    let term12  = c12 * inv_d12;

                    let force_scale = (6.0 * term6 - 12.0 * term12) * inv_d2;
                    fx += dx * force_scale;
                    fy += dy * force_scale;
                    fz += dz * force_scale;
                    pe += 0.5 * (term12 - term6);
                }
            }
        }
    }

    force[i].x += fx;
    force[i].y += fy;
    force[i].z += fz;
    energy[i]  += pe;
}
