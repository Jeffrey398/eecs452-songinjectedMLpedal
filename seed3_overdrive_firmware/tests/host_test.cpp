// host_test.cpp - runs the firmware's DSP and protocol code on a laptop.
//   host_test protocol
//   host_test dsp in.f32 out.f32 sr drive tone level
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../knob_protocol.h"
#include "../test_source.h"
#include "../overdrive_dsp.h"

using namespace od;

static int g_fail = 0;
#define CHECK(cond) do { if(!(cond)) { std::printf("  FAIL line %d: %s\n", __LINE__, #cond); g_fail++; } } while(0)
static bool Near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

static int RunProtocolTests()
{
    Knobs k;
    std::printf("parse: three values, two values, spaces, clamping\n");
    CHECK(ParseKnobLine("0.75,0.4,0.6", &k) == ParseResult::Ok);
    CHECK(Near(k.drive, 0.75f) && Near(k.tone, 0.4f) && Near(k.level, 0.6f));
    CHECK(ParseKnobLine(" 0.1 , 0.9 ", &k) == ParseResult::Ok);          // two values: level unchanged
    CHECK(Near(k.drive, 0.1f) && Near(k.tone, 0.9f) && Near(k.level, 0.6f));
    CHECK(ParseKnobLine("5,-3,2", &k) == ParseResult::Ok);
    CHECK(Near(k.drive, 1.0f) && Near(k.tone, 0.0f) && Near(k.level, 1.0f));
    CHECK(ParseKnobLine("1e-1,0,0.5", &k) == ParseResult::Ok && Near(k.drive, 0.1f));

    std::printf("parse: rejects bad lines and leaves the knobs untouched\n");
    Knobs keep; keep.drive = 0.33f; k = keep;
    CHECK(ParseKnobLine("", &k) == ParseResult::Empty);
    CHECK(ParseKnobLine("0.5", &k) == ParseResult::WrongCount);
    CHECK(ParseKnobLine("1,2,3,4", &k) == ParseResult::WrongCount);
    CHECK(ParseKnobLine("0.5,", &k) == ParseResult::WrongCount);
    CHECK(ParseKnobLine("a,b,c", &k) == ParseResult::BadNumber);
    CHECK(ParseKnobLine("0.5,0.5,0.5x", &k) == ParseResult::TrailingJunk);
    CHECK(ParseKnobLine("nan,0.5,0.5", &k) == ParseResult::NotFinite);
    CHECK(ParseKnobLine("0.5,inf,0.5", &k) == ParseResult::NotFinite);
    CHECK(Near(k.drive, 0.33f));

    std::printf("commands\n");
    CHECK(IsCommand("ping", "ping") && IsCommand("  PING  ", "ping") && IsCommand("Bypass", "bypass"));
    CHECK(!IsCommand("pingx", "ping") && !IsCommand("0.5,0.5,0.5", "ping") && !IsCommand("", "ping"));

    std::printf("line queue: chunks, CRLF, two lines per chunk, garbage, overflow\n");
    LineQueue q; char line[LineQueue::kLineMax];
    const char* c1 = "0.7,"; const char* c2 = "0.4,0.5"; const char* c3 = "\r\n0.1,0.2,0.3\n";
    q.Push((const uint8_t*)c1, std::strlen(c1));
    CHECK(!q.Pop(line, sizeof(line)));
    q.Push((const uint8_t*)c2, std::strlen(c2));
    q.Push((const uint8_t*)c3, std::strlen(c3));
    CHECK(q.Pop(line, sizeof(line)) && std::strcmp(line, "0.7,0.4,0.5") == 0);
    CHECK(q.Pop(line, sizeof(line)) && std::strcmp(line, "0.1,0.2,0.3") == 0);
    CHECK(!q.Pop(line, sizeof(line)));
    std::vector<uint8_t> junk(300, 'x'); const char* nl = "\n";
    q.Push(junk.data(), junk.size()); q.Push((const uint8_t*)nl, 1);
    CHECK(!q.Pop(line, sizeof(line)));
    const char* good = "0.2,0.2,0.2\n"; q.Push((const uint8_t*)good, std::strlen(good));
    CHECK(q.Pop(line, sizeof(line)) && std::strcmp(line, "0.2,0.2,0.2") == 0);
    for(int i = 0; i < 10; i++) { char m[32]; int n = std::snprintf(m, sizeof(m), "0.%d,0.5,0.5\n", i); q.Push((const uint8_t*)m, n); }
    int popped = 0; while(q.Pop(line, sizeof(line))) popped++;
    CHECK(popped == (int)LineQueue::kSlots - 1);

    std::printf("mailbox\n");
    KnobMailbox mb; Knobs w, r; w.drive = 0.9f; w.tone = 0.1f; mb.Write(w);
    CHECK(mb.TryRead(&r) && Near(r.drive, 0.9f) && Near(r.tone, 0.1f));

    std::printf("effect: not engaged = exact pass-through\n");
    Overdrive fx; fx.Init(48000.0f, 48);
    float maxerr = 0;
    for(int i = 0; i < 4800; i++) { float x = 0.5f * std::sin(2 * (float)kPi * 440.0f * i / 48000.0f); maxerr = std::fmax(maxerr, std::fabs(fx.Process(x) - x)); }
    CHECK(maxerr == 0.0f);

    std::printf("effect: engages without a click and stays bounded\n");
    const float test_drives[] = {0.1f, 1.0f};
    for(float d : test_drives)
    {
        Overdrive g; g.Init(48000.0f, 48); Knobs hot; hot.drive = d; hot.tone = 1.0f; hot.level = 1.0f;
        g.SetTarget(hot); g.SetEngaged(true);
        float peak = 0, prev = 0, first = 0, settled = 0;
        for(int b = 0; b < 400; b++)
            for(int i = 0; i < 48; i++)
            {
                if(i == 0) g.UpdateBlock();
                float y = g.Process(0.9f * std::sin(2 * (float)kPi * 196.0f * (b * 48 + i) / 48000.0f));
                float s = std::fabs(y - prev); prev = y;
                peak = std::fmax(peak, std::fabs(y));
                if(b < 40) first = std::fmax(first, s);
                if(b >= 300) settled = std::fmax(settled, s);
            }
        CHECK(peak <= 1.0f && peak > 0.2f);
        CHECK(first <= 1.5f * settled + 0.01f);   // fade-in no harsher than 1.5x the settled waveform (filter start-up adds a little)
    }

    {   // built-in test source: finite, audible, not too hot, moves through several notes
        TestSource ts; ts.Init(48000.0f);
        float peak = 0.0f, energy = 0.0f; bool finite = true;
        for(int i = 0; i < 48000 * 3; i++) { float v = ts.Process(); if(!std::isfinite(v)) finite = false; peak = std::fmax(peak, std::fabs(v)); energy += v * v; }
        CHECK(finite);
        CHECK(peak > 0.05f && peak < 0.5f);
        CHECK(energy > 1.0f);
        const bool has_cmd = IsCommand("test", "test") && IsCommand(" LIVE ", "live") && !IsCommand("test 1", "test");
        CHECK(has_cmd);
    }
    std::printf(g_fail ? "PROTOCOL/UNIT TESTS FAILED (%d)\n" : "protocol/unit tests passed\n", g_fail);
    return g_fail ? 1 : 0;
}

static int RunDsp(int argc, char** argv)
{
    if(argc != 8) { std::fprintf(stderr, "usage: host_test dsp in.f32 out.f32 sr drive tone level\n"); return 2; }
    FILE* fi = std::fopen(argv[2], "rb"); if(!fi) { std::perror("open input"); return 2; }
    std::vector<float> x; float s; while(std::fread(&s, sizeof(float), 1, fi) == 1) x.push_back(s); std::fclose(fi);
    const float sr = (float)std::atof(argv[4]);
    Knobs k; k.drive = (float)std::atof(argv[5]); k.tone = (float)std::atof(argv[6]); k.level = (float)std::atof(argv[7]);
    const size_t kBlock = 48;
    Overdrive fx; fx.Init(sr, kBlock); fx.SetTarget(k); fx.SetEngaged(true); fx.SnapToTarget();
    std::vector<float> y(x.size());
    for(size_t i = 0; i < x.size(); i += kBlock)
    {
        fx.UpdateBlock();
        const size_t n = (x.size() - i < kBlock) ? (x.size() - i) : kBlock;
        for(size_t j = 0; j < n; j++) y[i + j] = fx.Process(x[i + j]);
    }
    FILE* fo = std::fopen(argv[3], "wb"); if(!fo) { std::perror("open output"); return 2; }
    std::fwrite(y.data(), sizeof(float), y.size(), fo); std::fclose(fo);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc >= 2 && !std::strcmp(argv[1], "protocol")) return RunProtocolTests();
    if(argc >= 2 && !std::strcmp(argv[1], "dsp")) return RunDsp(argc, argv);
    std::fprintf(stderr, "usage: host_test protocol | host_test dsp in.f32 out.f32 sr drive tone level\n");
    return 2;
}
