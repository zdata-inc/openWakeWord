from pathlib import Path
from subprocess import run
from tqdm import tqdm

input_dir = Path('fma_small')
output_dir = Path('fma')
for path in tqdm(list(input_dir.glob('**/*.mp3'))):
    # ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
    output_path = output_dir / f'{path.stem}.wav'
    print(path)
    run(['ffmpeg', '-i', path, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', output_path])
    print(output_path)
