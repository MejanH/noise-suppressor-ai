from torch.utils.data import DataLoader
from df import enhance, init_df
from df.enhance import AudioDataset
from df.model import ModelParams
from df.io import load_audio, resample, save_audio
from loguru import logger
import torch
import sys
import glob
import os
import time


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [output_file]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = "./temp"

    torch.cuda.empty_cache()
    model, df_state, suffix = init_df()

    input_files = [input_file]
    df_sr = ModelParams().sr
    ds = AudioDataset(input_files, df_sr)
    loader = DataLoader(ds, num_workers=2, pin_memory=True)

    n_samples = len(ds)
    for i, (file, audio, audio_sr) in enumerate(loader):
        file = file[0]
        audio = audio.squeeze(0)
        progress = (i + 1) / n_samples * 100
        t0 = time.time()
        audio = enhance(model, df_state, audio, atten_lim_db=None)
        t1 = time.time()
        t_audio = audio.shape[-1] / df_sr
        t = t1 - t0
        rtf = t / t_audio
        fn = os.path.basename(file)
        p_str = f"{progress:2.0f}% | " if n_samples > 1 else ""
        logger.info(
            f"{p_str}Enhanced noisy audio file '{fn}' in {t:.2f}s (RT factor: {rtf:.3f})"
        )
        audio = resample(audio.to("cpu"), df_sr, audio_sr)
        save_audio(
            file, audio, sr=audio_sr, output_dir=output_dir, suffix=suffix, log=False
        )


if __name__ == "__main__":
    main()
