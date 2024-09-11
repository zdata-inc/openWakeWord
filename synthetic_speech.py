import riva.client
import riva.client.audio_io
import numpy as np
import wave
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


def apply_ssml(text: str, voice_rate: str = '100%',
               voice_pitch: str = 'default',
               voice_volume: str = 'default'):
    """
    Apply SSML tags to text (if enabled)
    """
    if not text:
        return None

    ssml_checks = False

    if isinstance(voice_rate, str) and voice_rate != 'default':
        ssml_checks = True
    elif isinstance(voice_rate, float) and voice_rate != 1.0:
        ssml_checks = True
    elif voice_pitch != 'default' or voice_volume != 'volume':
        ssml_checks = True

    if ssml_checks:
        text = f"<prosody rate='{voice_rate}' pitch='{voice_pitch}' volume='{voice_volume}'>{text}</prosody>"

    return "<speak>" + text + "</speak>"


riva_server = "deeplearner:50051"
auth = riva.client.Auth(uri=riva_server)
tts_service = riva.client.SpeechSynthesisService(auth)

df = pd.read_csv('raft_AXIS__final_v240821-gptq/sft.csv')

utterances = []
for i, row in df.iterrows():
    chat = json.loads(row.chat)
    utterances.extend([message['payload']['content'] for message in chat
                      if message['payload']['role'] == 'assistant'])

utterances = [utt for utt in utterances if not utt.startswith('\'s.:\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\.\n\\\\\\\\\\\\\\\\Walke')]

output_dir = Path('koda_audio')
output_dir.mkdir(exist_ok=True, parents=True)

voice: str = 'English-US.Female-1'  # 'English-US.Male-1'
language_code: str = 'en-US'
sample_rate_hz: int = 16000

def write_wav_file(filename, samples, sample_rate):
    with wave.open(str(filename), 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

def merge_samples(sample_list):
    return np.concatenate(sample_list)

for i, utterance in tqdm(list(enumerate(utterances))):
    print(utterance)

    if (output_dir / f'{i}.wav').exists():
        continue

    responses = tts_service.synthesize_online(
        utterance, voice, language_code, sample_rate_hz=sample_rate_hz)

    all_samples = []
    for response in responses:
        samples = np.frombuffer(response.audio, dtype=np.int16)
        all_samples.append(samples)

    merged_samples = merge_samples(all_samples)
    write_wav_file(output_dir / f'{i}.wav', merged_samples, sample_rate_hz)