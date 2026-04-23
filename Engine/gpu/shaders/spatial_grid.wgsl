// Построение пространственной сетки на GPU: count → scan → sort
//
// Буферы 
//   binding 0 — uniform  Params
//   binding 1 — pos      array<vec4f>          (read,  count/sort)
//   binding 2 — counts   array<atomic<u32>>    (r/w,   count / scan_blocks)
//   binding 3 — starts   array<u32>            (write, scan / add_offsets)
//   binding 4 — off      array<atomic<u32>>    (r/w,   add_offsets / sort)
//   binding 5 — sortedIdx array<u32>           (write, sort)
//   binding 6 — blockSums array<u32>           (r/w,   scan_blocks / scan_level2 / add_offsets)
//
// Dispatch-последовательность:
//   1. count        — workgroups: ceil(n / 64)
//   2. scan_blocks  — workgroups: ceil(totalCells / 256)
//   3. scan_level2  — workgroups: 1   (один workgroup на весь blockSums)
//   4. add_offsets  — workgroups: ceil(totalCells / 256)
//   5. sort         — workgroups: ceil(n / 64)
//
// Поддерживает до 256 * 256 = 65 536 ячеек.

struct Params {
    cellSize   : f32,
    dx         : u32,
    dy         : u32,
    dz         : u32,
    n          : u32,
    _pad       : u32,
};

@group(0) @binding(0) var<uniform>             p         : Params;
@group(0) @binding(1) var<storage, read>       pos       : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> counts    : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> starts    : array<u32>;
@group(0) @binding(4) var<storage, read_write> off       : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> sortedIdx : array<u32>;
@group(0) @binding(6) var<storage, read_write> blockSums : array<u32>;

fn cell_idx(v: vec3f) -> u32 {
    let cx = clamp(u32(v.x / p.cellSize), 0u, p.dx - 1u);
    let cy = clamp(u32(v.y / p.cellSize), 0u, p.dy - 1u);
    let cz = clamp(u32(v.z / p.cellSize), 0u, p.dz - 1u);
    return cx + cy * p.dx + cz * p.dx * p.dy;
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= p.n) { return; }
    atomicAdd(&counts[cell_idx(pos[gid.x].xyz)], 1u);
}

var<workgroup> ws_scan: array<u32, 256>;
@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(global_invocation_id) gid : vec3u,
    @builtin(local_invocation_id)  lid : vec3u,
    @builtin(workgroup_id)         wid : vec3u,
) {
    let total_cells = p.dx * p.dy * p.dz;

    // Загружаем counts в shared memory; за пределами — 0.
    ws_scan[lid.x] = select(0u, atomicLoad(&counts[gid.x]), gid.x < total_cells);
    workgroupBarrier();

    // Inclusive Hillis-Steele scan.
    // После цикла ws_scan[i] = sum(counts[blockStart .. blockStart+i])
    for (var stride = 1u; stride < 256u; stride <<= 1u) {
        let val = ws_scan[lid.x] + select(0u, ws_scan[lid.x - stride], lid.x >= stride);
        workgroupBarrier();
        ws_scan[lid.x] = val;
        workgroupBarrier();
    }

    // Сумма всего блока → blockSums (нужна для уровня 2).
    if (lid.x == 255u) {
        blockSums[wid.x] = ws_scan[255u];
    }

    // Конвертируем inclusive → exclusive и пишем в starts.
    if (gid.x < total_cells) {
        starts[gid.x] = select(0u, ws_scan[lid.x - 1u], lid.x > 0u);
    }
}

var<workgroup> ws_level2: array<u32, 256>;
@compute @workgroup_size(256)
fn scan_level2(@builtin(local_invocation_id) lid: vec3u) {
    let total_blocks = (p.dx * p.dy * p.dz + 255u) / 256u;

    ws_level2[lid.x] = select(0u, blockSums[lid.x], lid.x < total_blocks);
    workgroupBarrier();

    // Inclusive Hillis-Steele
    for (var stride = 1u; stride < 256u; stride <<= 1u) {
        let val = ws_level2[lid.x] + select(0u, ws_level2[lid.x - stride], lid.x >= stride);
        workgroupBarrier();
        ws_level2[lid.x] = val;
        workgroupBarrier();
    }

    if (lid.x < total_blocks) {
        blockSums[lid.x] = select(0u, ws_level2[lid.x - 1u], lid.x > 0u);
    }
}

@compute @workgroup_size(256)
fn add_offsets(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id)         wid: vec3u,
) {
    let total_cells = p.dx * p.dy * p.dz;
    if (gid.x >= total_cells) { return; }

    let final_offset = starts[gid.x] + blockSums[wid.x];
    starts[gid.x] = final_offset;

    atomicStore(&off[gid.x], final_offset);

    if (gid.x == total_cells - 1u) {
        starts[total_cells] = p.n;
    }
}

@compute @workgroup_size(64)
fn sort(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= p.n) { return; }
    let cell      = cell_idx(pos[gid.x].xyz);
    let write_pos = atomicAdd(&off[cell], 1u);
    sortedIdx[write_pos] = gid.x;
}