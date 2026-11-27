#pragma once

#include "bluesdriver_wh_params.h"

inline float Clamp01(float x)
{
    if(x < 0.0f) return 0.0f;
    if(x > 1.0f) return 1.0f;
    return x;
}

inline float Lerp(float a, float b, float w)
{
    return a + w * (b - a);
}

inline float Bilerp(float p00,
                    float p01,
                    float p10,
                    float p11,
                    float a,
                    float b)
{
    float low  = Lerp(p00, p01, b);
    float high = Lerp(p10, p11, b);

    return Lerp(low, high, a);
}

inline WHParams InterpolateWH(float drive, float tone)
{
    float g = Clamp01(drive) * 5.0f;
    float t = Clamp01(tone) * 5.0f;

    int g0 = static_cast<int>(g);
    int t0 = static_cast<int>(t);

    if(g0 >= 5) g0 = 4;
    if(t0 >= 5) t0 = 4;

    int g1 = g0 + 1;
    int t1 = t0 + 1;

    float a = g - g0;
    float b = t - t0;

    const WHParams& p00 = BLUESDRIVER_WH[g0][t0];
    const WHParams& p01 = BLUESDRIVER_WH[g0][t1];
    const WHParams& p10 = BLUESDRIVER_WH[g1][t0];
    const WHParams& p11 = BLUESDRIVER_WH[g1][t1];

    WHParams result;

    result.pre_hp = Bilerp(
        p00.pre_hp, p01.pre_hp,
        p10.pre_hp, p11.pre_hp, a, b);

    result.pre_lp = Bilerp(
        p00.pre_lp, p01.pre_lp,
        p10.pre_lp, p11.pre_lp, a, b);

    result.gain = Bilerp(
        p00.gain, p01.gain,
        p10.gain, p11.gain, a, b);

    result.bias = Bilerp(
        p00.bias, p01.bias,
        p10.bias, p11.bias, a, b);

    result.post_lp = Bilerp(
        p00.post_lp, p01.post_lp,
        p10.post_lp, p11.post_lp, a, b);

    result.output_gain = Bilerp(
        p00.output_gain, p01.output_gain,
        p10.output_gain, p11.output_gain, a, b);

    return result;
}
