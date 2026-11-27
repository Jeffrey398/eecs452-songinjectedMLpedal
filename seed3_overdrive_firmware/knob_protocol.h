// knob_protocol.h - the USB serial protocol between the Pi and the Seed3.
//
//   Pi -> Seed3 :  "drive,tone,level\n"   three numbers 0..1, e.g. "0.7500,0.4000,0.5000\n"
//                  "drive,tone\n"         two numbers also accepted (level unchanged)
//                  "bypass\n"             back to a clean pass-through
//                  "ping\n"               connectivity check
//   Seed3 -> Pi :  "OK d,t,l"             accepted values x1000 as integers
//                  "PONG" / "OK bypass"
//                  "ERR <reason>"         line rejected, previous settings kept
//
// Header-only and free of libDaisy, so it is tested on a laptop (tests/).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "overdrive_dsp.h"

namespace od {

enum class ParseResult { Ok, Empty, WrongCount, BadNumber, NotFinite, TrailingJunk };

inline const char* ParseResultName(ParseResult r)
{
    switch(r)
    {
        case ParseResult::Ok: return "ok";
        case ParseResult::Empty: return "empty";
        case ParseResult::WrongCount: return "wrong_count";
        case ParseResult::BadNumber: return "bad_number";
        case ParseResult::NotFinite: return "not_finite";
        case ParseResult::TrailingJunk: return "trailing_junk";
    }
    return "unknown";
}

inline const char* SkipSpace(const char* p)
{
    while(*p == ' ' || *p == '\t') ++p;
    return p;
}

// Parses two or three comma-separated numbers into *k (clamped to 0..1). With two numbers
// only drive and tone change. On failure *k is untouched.
inline ParseResult ParseKnobLine(const char* line, Knobs* k)
{
    float       v[3];
    int         n = 0;
    const char* p = SkipSpace(line);
    if(*p == '\0') return ParseResult::Empty;

    for(;;)
    {
        p = SkipSpace(p);
        char* end = nullptr;
        float f   = std::strtof(p, &end);
        if(end == p) return (*p == '\0') ? ParseResult::WrongCount : ParseResult::BadNumber;
        if(!std::isfinite(f)) return ParseResult::NotFinite;
        if(n == 3) return ParseResult::WrongCount;
        v[n++] = f;
        p      = SkipSpace(end);
        if(*p == ',') { ++p; continue; }
        break;
    }
    if(*p != '\0') return ParseResult::TrailingJunk;
    if(n < 2) return ParseResult::WrongCount;

    Knobs q = *k;
    q.drive = v[0];
    q.tone  = v[1];
    if(n == 3) q.level = v[2];
    ClampKnobs(&q);
    *k = q;
    return ParseResult::Ok;
}

// Case-insensitive match against a command word (no allocation, no libc extras).
inline bool IsCommand(const char* line, const char* word)
{
    line = SkipSpace(line);
    while(*word)
    {
        char a = *line, b = *word;
        if(a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if(a != b) return false;
        ++line; ++word;
    }
    return *SkipSpace(line) == '\0';
}

// USB byte stream -> complete lines. One producer (USB receive interrupt), one consumer (main loop).
// An over-long line is discarded; a full queue drops the newest line.
class LineQueue
{
  public:
    static constexpr size_t kLineMax = 96;
    static constexpr size_t kSlots   = 4;

    void Push(const uint8_t* data, size_t n)
    {
        for(size_t i = 0; i < n; i++)
        {
            const char c = static_cast<char>(data[i]);
            if(c == '\n')
            {
                if(!overflow_ && len_ > 0) Enqueue();
                len_ = 0;
                overflow_ = false;
            }
            else if(c == '\r') { /* ignore so CRLF works */ }
            else if(len_ < kLineMax - 1) cur_[len_++] = c;
            else overflow_ = true;
        }
    }

    bool Pop(char* out, size_t cap)
    {
        if(tail_ == head_) return false;
        size_t i = 0;
        for(; i < cap - 1 && slots_[tail_][i] != '\0'; i++) out[i] = slots_[tail_][i];
        out[i] = '\0';
        tail_  = (tail_ + 1) % kSlots;
        return true;
    }

  private:
    void Enqueue()
    {
        const size_t next = (head_ + 1) % kSlots;
        if(next == tail_) return;
        for(size_t i = 0; i < len_; i++) slots_[head_][i] = cur_[i];
        slots_[head_][len_] = '\0';
        head_ = next;
    }

    char            slots_[kSlots][kLineMax] = {};
    char            cur_[kLineMax]           = {};
    size_t          len_                     = 0;
    bool            overflow_                = false;
    volatile size_t head_                    = 0;
    volatile size_t tail_                    = 0;
};

// Lock-free hand-off of the latest Knobs from the main loop to the audio interrupt.
// The audio callback must never wait, so TryRead() gives up if a write is in progress.
class KnobMailbox
{
  public:
    void Write(const Knobs& k)
    {
        seq_ = seq_ + 1;
        Barrier();
        data_ = k;
        Barrier();
        seq_ = seq_ + 1;
    }
    bool TryRead(Knobs* out) const
    {
        const uint32_t s1 = seq_;
        if(s1 & 1u) return false;
        Barrier();
        Knobs copy = data_;
        Barrier();
        if(seq_ != s1) return false;
        *out = copy;
        return true;
    }

  private:
    static void Barrier() { __asm__ __volatile__("" ::: "memory"); }
    volatile uint32_t seq_ = 0;
    Knobs             data_;
};

} // namespace od
