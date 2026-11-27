// main.cpp - Seed3 overdrive pedal: Wiener-Hammerstein model driven over USB serial.
//
//   guitar in -> [H1 -> nonlinearity -> H2 -> level] -> out (both channels)
//   Raspberry Pi --USB--> "drive,tone,level\n" --> new knob settings
//
// Starts as a CLEAN PASS-THROUGH. The effect engages when the first valid knob line arrives
// ("bypass" returns to clean). Parameters come only from USB; pots are not read yet.
//
// "test" plays a built-in guitar-like signal (no input cable needed); "live" returns to the input jack.
//
// Build: make        Flash: Web Programmer with build/seed3_overdrive.bin (Seed in bootloader mode)
#include "daisy_seed.h"

#include "knob_protocol.h"
#include "overdrive_dsp.h"
#include "test_source.h"

using namespace daisy;

// ---- settings to confirm against the board ----------------------------------
constexpr size_t kBlockSize    = 48;   // samples per audio callback
constexpr size_t kInputChannel = 0;    // 0 = left input jack (guitar); 1 = right
// ------------------------------------------------------------------------------

static DaisySeed       hw;
static od::Overdrive   fx;
static od::KnobMailbox mailbox;
static od::LineQueue   line_queue;
static od::TestSource  testsrc;
static volatile bool   g_engaged = false;
static volatile bool   g_test    = false;   // true = play the built-in test signal instead of the input jack

// USB receive callback: interrupt context, so it only queues bytes.
static void UsbReceive(uint8_t* buf, uint32_t* len)
{
    if(buf && len) line_queue.Push(buf, *len);
}

static void AudioCallback(AudioHandle::InputBuffer  in,
                          AudioHandle::OutputBuffer out,
                          size_t                    size)
{
    od::Knobs k;
    if(mailbox.TryRead(&k)) fx.SetTarget(k);
    fx.SetEngaged(g_engaged);
    fx.UpdateBlock();

    for(size_t i = 0; i < size; i++)
    {
        const float x = g_test ? testsrc.Process() : in[kInputChannel][i];
        const float y = fx.Process(x);
        out[0][i]     = y;
        out[1][i]     = y;
    }
}

static int Milli(float v) { return static_cast<int>(lroundf(v * 1000.0f)); }

int main(void)
{
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.SetAudioBlockSize(kBlockSize);

    od::Knobs knobs; // defaults 0.5 / 0.5 / 0.5
    fx.Init(hw.AudioSampleRate(), hw.AudioBlockSize());
    testsrc.Init(hw.AudioSampleRate());
    fx.SetTarget(knobs);
    fx.SnapToTarget();
    mailbox.Write(knobs);

    // USB serial. false = do not wait for a PC, so the pedal passes audio with nothing connected.
    hw.StartLog(false);
    hw.usb_handle.SetReceiveCallback(UsbReceive, UsbHandle::UsbPeriph::FS_INTERNAL);
    hw.StartAudio(AudioCallback);

    bool led = false;
    char line[od::LineQueue::kLineMax];
    while(1)
    {
        while(line_queue.Pop(line, sizeof(line)))
        {
            if(od::IsCommand(line, "ping")) { hw.PrintLine("PONG"); continue; }
            if(od::IsCommand(line, "test")) { g_test = true;  hw.PrintLine("OK test"); continue; }
            if(od::IsCommand(line, "live")) { g_test = false; hw.PrintLine("OK live"); continue; }
            if(od::IsCommand(line, "bypass"))
            {
                g_engaged = false;
                hw.PrintLine("OK bypass");
                continue;
            }
            const od::ParseResult r = od::ParseKnobLine(line, &knobs);
            if(r == od::ParseResult::Ok)
            {
                mailbox.Write(knobs);
                g_engaged = true;
                led = !led;
                hw.SetLed(led);
                hw.PrintLine("OK %d,%d,%d", Milli(knobs.drive), Milli(knobs.tone), Milli(knobs.level));
            }
            else
            {
                hw.PrintLine("ERR %s", od::ParseResultName(r));
            }
        }
    }
}
