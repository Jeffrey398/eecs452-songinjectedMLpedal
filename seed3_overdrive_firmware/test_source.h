// test_source.h - built-in guitar-like test signal, so the pedal can be heard with no input cable.
// Plays a repeating plucked line (E2, A2, D3, A2: sawtooth + soft low-pass, exponential decay).
// Header-only, no libDaisy dependency, tested on a laptop.
#pragma once
#include <cmath>

namespace od {

class TestSource
{
  public:
    void Init(float sr) { sr_ = sr; Reset(); }
    void Reset() { phase_ = 0.0f; t_ = 0.0f; note_ = 0; lp_ = 0.0f; }

    float Process()
    {
        static const float kNotes[4] = {82.41f, 110.0f, 146.83f, 110.0f};
        const float kNoteSec = 0.6f;
        const float f        = kNotes[note_];
        phase_ += f / sr_;
        if(phase_ >= 1.0f) phase_ -= 1.0f;
        const float saw = 2.0f * phase_ - 1.0f;
        lp_ += 0.25f * (saw - lp_);               // soften the sawtooth (~2 kHz)
        const float env = std::exp(-3.0f * t_ / kNoteSec);
        t_ += 1.0f / sr_;
        if(t_ >= kNoteSec) { t_ = 0.0f; note_ = (note_ + 1) & 3; }
        return 0.4f * env * lp_;                  // peak well below full scale
    }

  private:
    float sr_ = 48000.0f, phase_ = 0.0f, t_ = 0.0f, lp_ = 0.0f;
    int   note_ = 0;
};

} // namespace od
