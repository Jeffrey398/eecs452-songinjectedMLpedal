// overdrive_dsp.h - Wiener-Hammerstein overdrive for the Seed3, driven by three knobs.
//
//        x --> [ H1: input filter ] --> u --> [ f: static nonlinearity ] --> v --> [ H2: output filter ] --> [ level ] --> y
//
//   H1  : high-pass + mid hump           (2 biquads)
//   f   : y = (tanh(g*u + b) - tanh(b)) / tanh(g)     g = drive gain, b = small asymmetry
//   H2  : tone low-pass + DC blocker     (1 biquad + 1 one-pole)
//   level: output gain
//
// THE NUMBERS BELOW ARE PLACEHOLDERS, NOT FITTED TO THE REAL PEDAL. The structure,
// the knob interface and the real-time plumbing are final; MapKnobs() is the one place
// where fitted values go. Replace its formulas (or make it interpolate a table measured
// on the 5x5 drive/tone grid) with the parameters identified from the recordings.
//
// Header-only, no libDaisy dependency, so it can be tested on a laptop (tests/).
#pragma once

#include <cmath>
#include <cstddef>

namespace od {

// All three knobs are normalised 0..1, as sent by the Pi.
struct Knobs
{
    float drive = 0.5f;
    float tone  = 0.5f;
    float level = 0.5f;
};

inline float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

inline void ClampKnobs(Knobs* k)
{
    k->drive = Clamp01(k->drive);
    k->tone  = Clamp01(k->tone);
    k->level = Clamp01(k->level);
}

// Everything the Wiener-Hammerstein model needs for one knob setting.
struct WhParams
{
    // H1: input filter
    double h1_hp_hz   = 80.0;     // high-pass corner
    double h1_mid_hz  = 720.0;    // mid hump centre
    double h1_mid_db  = 4.0;
    double h1_mid_q   = 0.8;
    // f: static nonlinearity
    float  nl_gain    = 1.0f;     // g
    float  nl_bias    = 0.0f;     // b
    // H2: output filter
    double h2_lp_hz   = 4000.0;   // tone low-pass corner
    // level
    float  out_gain   = 1.0f;
};

// PLACEHOLDER knob -> model mapping. Replace with fitted values.
inline WhParams MapKnobs(const Knobs& k)
{
    WhParams p;
    p.nl_gain  = 1.0f + 49.0f * k.drive * k.drive;                  // 1 .. 50
    p.nl_bias  = 0.10f * k.drive;                                   // slight asymmetry
    p.h2_lp_hz = 800.0 * std::pow(2.0, 3.5 * (double)k.tone);        // 800 Hz .. ~9 kHz
    p.out_gain = std::pow(10.0f, (k.level - 0.5f) * 18.0f / 20.0f);  // -9 .. +9 dB
    return p;
}

constexpr double kPi = 3.14159265358979323846;

// Direct form II transposed biquad, a0 normalised to 1. Coefficients may change while running.
struct Biquad
{
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
    void Set(double nb0, double nb1, double nb2, double na0, double na1, double na2)
    {
        b0 = (float)(nb0 / na0); b1 = (float)(nb1 / na0); b2 = (float)(nb2 / na0);
        a1 = (float)(na1 / na0); a2 = (float)(na2 / na0);
    }
    float Process(float x)
    {
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

// RBJ cookbook designs (Q = 1/sqrt(2) for the Butterworth-style high/low-pass)
inline void DesignHighPass(Biquad* f, double fs, double f0, double q)
{
    const double w = 2 * kPi * f0 / fs, cw = std::cos(w), alpha = std::sin(w) / (2 * q);
    f->Set((1 + cw) / 2, -(1 + cw), (1 + cw) / 2, 1 + alpha, -2 * cw, 1 - alpha);
}
inline void DesignLowPass(Biquad* f, double fs, double f0, double q)
{
    const double w = 2 * kPi * f0 / fs, cw = std::cos(w), alpha = std::sin(w) / (2 * q);
    f->Set((1 - cw) / 2, 1 - cw, (1 - cw) / 2, 1 + alpha, -2 * cw, 1 - alpha);
}
inline void DesignPeaking(Biquad* f, double fs, double f0, double gain_db, double q)
{
    const double A = std::pow(10.0, gain_db / 40.0), w = 2 * kPi * f0 / fs;
    const double cw = std::cos(w), alpha = std::sin(w) / (2 * q);
    f->Set(1 + alpha * A, -2 * cw, 1 - alpha * A, 1 + alpha / A, -2 * cw, 1 - alpha / A);
}

constexpr double kButterQ  = 0.70710678118654752;
constexpr float  kSmoothMs = 20.0f;   // knob glide, avoids zipper noise
constexpr float  kDcR      = 0.998f;  // output DC blocker pole (~15 Hz at 48 kHz)

class Overdrive
{
  public:
    void Init(float sample_rate, size_t block_size)
    {
        sr_ = sample_rate;
        glide_ = 1.0f - std::exp(-(static_cast<float>(block_size) / sr_) / (kSmoothMs / 1000.0f));
        SnapToTarget();
    }

    void SetTarget(const Knobs& k) { target_ = k; }

    // false = clean pass-through (start-up, and the BYPASS command), true = effect on.
    void SetEngaged(bool on) { wet_target_ = on ? 1.0f : 0.0f; }

    // Jump straight to the targets (start-up and tests that need exact values).
    void SnapToTarget()
    {
        cur_ = target_;
        wet_ = wet_target_;
        Derive();
    }

    // Call once per audio block, before Process().
    void UpdateBlock()
    {
        cur_.drive += glide_ * (target_.drive - cur_.drive);
        cur_.tone  += glide_ * (target_.tone - cur_.tone);
        cur_.level += glide_ * (target_.level - cur_.level);
        wet_       += glide_ * (wet_target_ - wet_);
        Derive();
    }

    float Process(float x)
    {
        // H1
        float u = h1_mid_.Process(h1_hp_.Process(x));
        // f
        float v = (std::tanh(p_.nl_gain * u + p_.nl_bias) - bias_t_) * nl_norm_;
        // H2
        float w = h2_lp_.Process(v);
        float y = w - dc_x1_ + kDcR * dc_y1_;
        dc_x1_ = w;
        dc_y1_ = y;
        y *= p_.out_gain;
        y = y > 1.0f ? 1.0f : (y < -1.0f ? -1.0f : y);
        // dry/wet crossfade lets the effect engage and disengage without a click
        return x + wet_ * (y - x);
    }

  private:
    void Derive()
    {
        p_ = MapKnobs(cur_);
        DesignHighPass(&h1_hp_, sr_, p_.h1_hp_hz, kButterQ);
        DesignPeaking(&h1_mid_, sr_, p_.h1_mid_hz, p_.h1_mid_db, p_.h1_mid_q);
        DesignLowPass(&h2_lp_, sr_, p_.h2_lp_hz, kButterQ);
        bias_t_  = std::tanh(p_.nl_bias);
        nl_norm_ = 1.0f / std::tanh(p_.nl_gain);
    }

    float    sr_ = 48000.0f, glide_ = 1.0f;
    Knobs    target_, cur_;
    WhParams p_;
    Biquad   h1_hp_, h1_mid_, h2_lp_;
    float    bias_t_ = 0.0f, nl_norm_ = 1.0f;
    float    dc_x1_ = 0.0f, dc_y1_ = 0.0f;
    float    wet_ = 0.0f, wet_target_ = 0.0f;
};

} // namespace od
