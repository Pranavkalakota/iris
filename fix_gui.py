content = open(r'C:\audio_stream\gui_phase9.py', 'r', encoding='utf-8').read()
old = 'recording_*.wav'
new = '*.wav'
count = content.count(old)
content = content.replace(old, new)
open(r'C:\audio_stream\gui_phase9.py', 'w', encoding='utf-8').write(content)
print(f'Fixed {count} occurrences')
