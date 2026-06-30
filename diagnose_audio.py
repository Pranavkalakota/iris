"""
Run this on Humza's laptop:  python diagnose_audio.py
It prints every host API and every input device with full details.
"""
import sounddevice as sd
import pyaudio

print("=" * 60)
print("SOUNDDEVICE HOST APIs")
print("=" * 60)
for i, h in enumerate(sd.query_hostapis()):
    print(f"  [{i}] {h['name']}  default_input={h['default_input_device']}")

print()
print("=" * 60)
print("SOUNDDEVICE INPUT DEVICES")
print("=" * 60)
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        api_name = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  [{i}] {d['name']}")
        print(f"       hostapi={d['hostapi']} ({api_name})")
        print(f"       max_input_ch={d['max_input_channels']}  default_sr={d['default_samplerate']}")

print()
print("=" * 60)
print("PYAUDIO HOST APIs")
print("=" * 60)
pa = pyaudio.PyAudio()
for i in range(pa.get_host_api_count()):
    info = pa.get_host_api_info_by_index(i)
    print(f"  [{i}] {info['name']}  deviceCount={info['deviceCount']}")

print()
print("=" * 60)
print("PYAUDIO INPUT DEVICES")
print("=" * 60)
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if int(info.get("maxInputChannels", 0)) > 0:
        api = pa.get_host_api_info_by_index(info["hostApi"])
        print(f"  [{i}] {info['name']}")
        print(f"       hostApi={info['hostApi']} ({api['name']})")
        print(f"       maxInputCh={info['maxInputChannels']}  defaultSR={info['defaultSampleRate']}")

pa.terminate()

print()
print("=" * 60)
print("WASAPI OPEN TEST (device 10, 44100 Hz)")
print("=" * 60)
try:
    wasapi = sd.WasapiSettings(exclusive=False)
    with sd.InputStream(device=10, samplerate=44100, channels=1,
                        dtype="int16", blocksize=2048,
                        extra_settings=wasapi) as s:
        print("  SUCCESS — WASAPI shared mode opened on device 10")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("WASAPI OPEN TEST (device 10, no extra_settings)")
try:
    with sd.InputStream(device=10, samplerate=44100, channels=1,
                        dtype="int16", blocksize=2048) as s:
        print("  SUCCESS — plain open on device 10")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("PLAIN OPEN TEST (device 9, 48000 Hz, no extra_settings)")
try:
    with sd.InputStream(device=9, samplerate=48000, channels=1,
                        dtype="int16", blocksize=2048) as s:
        print("  SUCCESS — plain open on device 9")
except Exception as e:
    print(f"  FAILED: {e}")