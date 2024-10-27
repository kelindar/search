//go:build ignore
// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

#include <stdint.h>
#include <math.h>

void f32_cosine_distance(const float *x, const float *y, double *result, const uint64_t size) {
    float sum_xy = 0.0f;
    float sum_xx = 0.0f;
    float sum_yy = 0.0f;

    #pragma clang loop vectorize(enable) interleave_count(2)
    for (uint64_t i = 0; i < size; i++) {
        sum_xy += x[i] * y[i];    // Sum of x * y
        sum_xx += x[i] * x[i];     // Sum of x * x
        sum_yy += y[i] * y[i];     // Sum of y * y
    }

    // Calculate the final result
    float denominator = sqrtf(sum_xx) * sqrtf(sum_yy);
    if (denominator == 0.0f) {
        *result = (double)0.0f;
        return;
    }

    double cosine_similarity = (double)sum_xy / (double)denominator;
    *result = cosine_similarity;
}