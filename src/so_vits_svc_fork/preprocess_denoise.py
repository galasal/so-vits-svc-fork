from logging import getLogger
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import os
import soundfile as sf
import shutil

LOG = getLogger(__name__)

def _process_one(
    input_path: Path,
    output_dir: Path,
    segment_seconds: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / f"{input_path.stem}.wav")

    audio, sr = librosa.load(input_path, sr=None, mono=True)
    samples_per_segment = int(segment_seconds * sr)
    #pad audio to be divisible by samples_per_segment
    length_after_padding = int(np.ceil(len(audio) / samples_per_segment)) * samples_per_segment
    padding_length = length_after_padding - len(audio)
    audio = librosa.util.fix_length(audio, length_after_padding)
    segments = librosa.util.frame(audio, frame_length=samples_per_segment, hop_length=samples_per_segment)
    processed_segments = []

    for i in range(segments.shape[1]):
        segment = segments[:, i]
        #remove padding if last segment
        if i == segments.shape[1] - 1:
            segment = segment[:len(segment) - padding_length]

        #demucs works with files, so need to write segment to file
        tmp_segment_file = (output_dir / "segment.wav")
        sf.write(tmp_segment_file, segment, sr)
        os.system(f"demucs --two-stems=vocals -o \"{output_dir}\" \"{tmp_segment_file}\"")
        tmp_separated_file = (output_dir / "htdemucs" / "segment" / "vocals.wav")
        separated, sr = librosa.load(tmp_separated_file, sr=None, mono=True)
        processed_segments.append(separated)
        demucs_output_dir = (output_dir / "htdemucs")
        #delete segment and demucs output files
        os.remove(tmp_segment_file)
        shutil.rmtree(demucs_output_dir)

    processed_signal = np.concatenate(processed_segments)
    sf.write(output_path, processed_signal, sr)


def preprocess_denoise(
        input_dir: Path | str,
        output_dir: Path | str,
        segment_seconds: int
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(input_dir.rglob("*.*"))

    with tqdm(desc="Splitting", total=len(input_paths)) as pbar:
        for input_path in input_paths:
            _process_one(
                input_path,
                output_dir / input_path.relative_to(input_dir).parent,
                segment_seconds,
            )
            pbar.update(1)