//go:build ignore
// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

#include <stdint.h>
#include <immintrin.h>
#include <math.h>

void f32_cosine_distance(const float *x, const float *y, const uint64_t size, float *result) {
    __m256 sum_xy = _mm256_setzero_ps();   // Sum of x * y
    __m256 sum_xx = _mm256_setzero_ps();   // Sum of x * x
    __m256 sum_yy = _mm256_setzero_ps();   // Sum of y * y

    uint64_t i;
    for (i = 0; i + 7 < size; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);

        sum_xy = _mm256_add_ps(sum_xy, _mm256_mul_ps(x_vec, y_vec));
        sum_xx = _mm256_add_ps(sum_xx, _mm256_mul_ps(x_vec, x_vec));
        sum_yy = _mm256_add_ps(sum_yy, _mm256_mul_ps(y_vec, y_vec));
    }

    // Manually inlining the sum of elements in each 256-bit vector
    // For sum_xy
    __m128 vlow_xy  = _mm256_castps256_ps128(sum_xy);
    __m128 vhigh_xy = _mm256_extractf128_ps(sum_xy, 1);
    vlow_xy  = _mm_add_ps(vlow_xy, vhigh_xy);
    __m128 shuf_xy = _mm_movehdup_ps(vlow_xy);
    __m128 sums_xy = _mm_add_ps(vlow_xy, shuf_xy);
    shuf_xy = _mm_movehl_ps(shuf_xy, sums_xy);
    sums_xy = _mm_add_ss(sums_xy, shuf_xy);
    float dot_xy = _mm_cvtss_f32(sums_xy);

    // For sum_xx
    __m128 vlow_xx  = _mm256_castps256_ps128(sum_xx);
    __m128 vhigh_xx = _mm256_extractf128_ps(sum_xx, 1);
    vlow_xx  = _mm_add_ps(vlow_xx, vhigh_xx);
    __m128 shuf_xx = _mm_movehdup_ps(vlow_xx);
    __m128 sums_xx = _mm_add_ps(vlow_xx, shuf_xx);
    shuf_xx = _mm_movehl_ps(shuf_xx, sums_xx);
    sums_xx = _mm_add_ss(sums_xx, shuf_xx);
    float norm_x = _mm_cvtss_f32(sums_xx);

    // For sum_yy
    __m128 vlow_yy  = _mm256_castps256_ps128(sum_yy);
    __m128 vhigh_yy = _mm256_extractf128_ps(sum_yy, 1);
    vlow_yy  = _mm_add_ps(vlow_yy, vhigh_yy);
    __m128 shuf_yy = _mm_movehdup_ps(vlow_yy);
    __m128 sums_yy = _mm_add_ps(vlow_yy, shuf_yy);
    shuf_yy = _mm_movehl_ps(shuf_yy, sums_yy);
    sums_yy = _mm_add_ss(sums_yy, shuf_yy);
    float norm_y = _mm_cvtss_f32(sums_yy);

    // Handle any remaining elements
    for (; i < size; i++) {
        dot_xy += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    float denominator = sqrtf(norm_x) * sqrtf(norm_y);

    // Avoid division by zero
    if (denominator == 0.0f) {
        *result = 0.0f;
        return;
    }

    float cosine_similarity = dot_xy / denominator;
    float cosine_distance = 1.0f - cosine_similarity;

    *result = cosine_distance;
}