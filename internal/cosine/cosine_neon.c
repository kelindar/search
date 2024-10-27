//go:build ignore
// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

#include <stdint.h>
#include <arm_neon.h>
#include <math.h>

void f32_cosine_distance(const float *x, const float *y, double *result, const uint64_t size) {
    float32x4_t sum_xy = vdupq_n_f32(0.0f);   // Sum of x * y
    float32x4_t sum_xx = vdupq_n_f32(0.0f);   // Sum of x * x
    float32x4_t sum_yy = vdupq_n_f32(0.0f);   // Sum of y * y

    uint64_t i;
    for (i = 0; i + 3 < size; i += 4) {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);

        sum_xy = vmlaq_f32(sum_xy, x_vec, y_vec);
        sum_xx = vmlaq_f32(sum_xx, x_vec, x_vec);
        sum_yy = vmlaq_f32(sum_yy, y_vec, y_vec);
    }

    // Sum the elements of the vectors
    float dot_xy = vaddvq_f32(sum_xy);
    float norm_x = vaddvq_f32(sum_xx);
    float norm_y = vaddvq_f32(sum_yy);

    // Handle any remaining elements
    for (; i < size; i++) {
        dot_xy += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }


    // Avoid division by zero
    float denominator = sqrtf(norm_x) * sqrtf(norm_y);
    if (denominator == 0.0f) {
        *result = (double)0.0f;
        return;
    }

    double cosine_similarity = (double)dot_xy / (double)denominator;
    *result = cosine_similarity;
}