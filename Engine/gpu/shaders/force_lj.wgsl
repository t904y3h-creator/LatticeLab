// Биндинги:
//   @group(0)
//     0 — uniform  LJParams
//     1 — storage  pos        array<vec4f>    (r)
//     2 — storage  force      array<vec4f>    (rw)
//     3 — storage  energy     array<f32>      (rw)
//     4 — storage  nl_offsets array<u32>      (r)  — CSR offsets полного списка
//     5 — storage  nl_nbrs    array<u32>      (r)  — CSR values  полного списка
//     6 — storage  lj_table   array<vec2f>    (r)  — [C6, C12] для каждой пары типов
//     7 — storage  atom_types array<u32>      (r)  — тип каждого атома

struct LJUniforms {
    atom_count  : u32,
    type_count  : u32,
    cutoff_sqr  : f32,

    grid_dim_x  : u32,
    grid_dim_y  : u32,
    grid_dim_z  : u32,
    cell_size   : f32,

    _pad        : u32,
};

@group(0) @binding(0) var<uniform>             u              : LJUniforms;
@group(0) @binding(1) var<storage, read>       pos            : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> force          : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> energy         : array<f32>;
@group(0) @binding(4) var<storage, read>       grid_offsets   : array<u32>;
@group(0) @binding(5) var<storage, read>       sorted_indices : array<u32>;
@group(0) @binding(6) var<storage, read>       lj_table       : array<vec2f>; // .x=C6, .y=C12
@group(0) @binding(7) var<storage, read>       atom_types     : array<u32>;

fn cell_idx(v: vec3f) -> u32 {
    let cx = clamp(u32(v.x / u.cellSize), 0u, u.grid_dim_x - 1u);
    let cy = clamp(u32(v.y / u.cellSize), 0u, u.grid_dim_y - 1u);
    let cz = clamp(u32(v.z / u.cellSize), 0u, u.grid_dim_z - 1u);
    return cx + cy * u.grid_dim_x + cz * u.grid_dim_x * u.grid_dim_y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.atom_count) { return; }

    let pi      = pos[i].xyz;
    let type_i  = atom_types[i];
    let row_off = type_i * u.type_count; // начало строки типа i в lj_table

    var fx = 0.0f;
    var fy = 0.0f;
    var fz = 0.0f;
    var pe = 0.0f;

    let cell_c = cell_idx(pi);

    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor_c = cell_c + vec3i(x, y, z);
                if (any(neighbor_c < vec3i(0)) || 
                    any(neighbor_c >= vec3i(i32(u.grid_dim_x), i32(u.grid_dim_y), i32(u.grid_dim_z)))) {
                    continue;
                }

                let cell_idx = u32(neighbor_c.x) + 
                               u32(neighbor_c.y) * u.grid_dim_x + 
                               u32(neighbor_c.z) * u.grid_dim_x * u.grid_dim_y;

                let begin = grid_offsets[cell_idx];
                let end   = grid_offsets[cell_idx + 1u];

                for (var p = begin; p < end; p++) {
                    let j = sorted_indices[p];
                    if (i == j) { continue; }

                    let pj = pos[j].xyz;

                    let dx = pj.x - pi.x;
                    let dy = pj.y - pi.y;
                    let dz = pj.z - pi.z;
                    let d2 = dx * dx + dy * dy + dz * dz;

                    if (d2 <= 0.0 || d2 > u.cutoff_sqr) { continue; }

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

                    // Потенциал: каждый атом накапливает свою половину пары.
                    // В full-списке пара (i,j) присутствует дважды — делим на 2.
                    pe += 0.5 * (term12 - term6);
                }
            }
        }
    }

    force[i].x += fx;
    force[i].y += fy;
    force[i].z += fz;
    energy[i] += pe;
}
