#include <metal_stdlib>
using namespace metal;

constant float RADIUS_SCALE [[function_constant(0)]];
constant float ANGLE_SCALE [[function_constant(1)]];

kernel void cartesian_to_polar_layer(
    device const float* in_data [[buffer(0)]],
    device float* radius_out [[buffer(1)]],
    device float* angle_out [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    float even = in_data[index * 2];
    float odd  = in_data[index * 2 + 1];
    
    radius_out[index] = sqrt(even * even + odd * odd);
    angle_out[index]  = atan2(odd, even);
}

constant float QJL_CORRECTION [[function_constant(2)]];

kernel void polar_to_cartesian_fused_3bit_qjl(
    device const uchar* radius_idx [[buffer(0)]],
    device const uchar* angle_idx  [[buffer(1)]],
    device const uchar* qjl_mask   [[buffer(2)]],
    device float* out_data [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    float r = (float)radius_idx[index] * RADIUS_SCALE;
    float a = (float)angle_idx[index] * ANGLE_SCALE;
    
    // 1-bit QJL Residual Correction
    float correction = (qjl_mask[index] > 0) ? QJL_CORRECTION : -QJL_CORRECTION;
    r += correction;
    
    out_data[index * 2]     = r * cos(a);
    out_data[index * 2 + 1] = r * sin(a);
}
