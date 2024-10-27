//go:build ignore
// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.


#include <stdint.h>
#include <immintrin.h>
#include <math.h>

void f32_cosine_distance(const float *x, const float *y, double *result, const uint64_t size) {
    __m256 sum_xy = _mm256_setzero_ps();   // Sum of x * y
    __m256 sum_xx = _mm256_setzero_ps();   // Sum of x * x
    __m256 sum_yy = _mm256_setzero_ps();   // Sum of y * y

    uint64_t i;
    for (i = 0; i <= size - 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);

        sum_xy = _mm256_fmadd_ps(x_vec, y_vec, sum_xy); // sum_xy += x_vec * y_vec
        sum_xx = _mm256_fmadd_ps(x_vec, x_vec, sum_xx); // sum_xx += x_vec * x_vec
        sum_yy = _mm256_fmadd_ps(y_vec, y_vec, sum_yy); // sum_yy += y_vec * y_vec
    }

    // Sum elements of sum_xy
    __m256 temp_xy = _mm256_hadd_ps(sum_xy, sum_xy);   // Sum adjacent pairs
    temp_xy = _mm256_hadd_ps(temp_xy, temp_xy);        // Sum adjacent quadruples
    __m128 sum_xy_128 = _mm_add_ps(_mm256_castps256_ps128(temp_xy), _mm256_extractf128_ps(temp_xy, 1));
    float dot_xy = _mm_cvtss_f32(sum_xy_128);          // Extract final sum

    // Sum elements of sum_xx
    __m256 temp_xx = _mm256_hadd_ps(sum_xx, sum_xx);
    temp_xx = _mm256_hadd_ps(temp_xx, temp_xx);
    __m128 sum_xx_128 = _mm_add_ps(_mm256_castps256_ps128(temp_xx), _mm256_extractf128_ps(temp_xx, 1));
    float norm_x = _mm_cvtss_f32(sum_xx_128);

    // Sum elements of sum_yy
    __m256 temp_yy = _mm256_hadd_ps(sum_yy, sum_yy);
    temp_yy = _mm256_hadd_ps(temp_yy, temp_yy);
    __m128 sum_yy_128 = _mm_add_ps(_mm256_castps256_ps128(temp_yy), _mm256_extractf128_ps(temp_yy, 1));
    float norm_y = _mm_cvtss_f32(sum_yy_128);

    // Handle remaining elements (if any)
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

    double cosine_similarity = (double)dot_xy / (double)denominator;
    *result = cosine_similarity;
}