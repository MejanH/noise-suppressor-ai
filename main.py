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
import subprocess
import tempfile
from pathlib import Path


def is_video_file(file_path):
    """Check if file is a video based on extension"""
    video_extensions = {
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".3gp",
    }
    return Path(file_path).suffix.lower() in video_extensions


def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video using ffmpeg"""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM 16-bit
            "-ar",
            "48000",  # Sample rate
            "-ac",
            "2",  # Stereo
            "-y",  # Overwrite output
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Audio extracted from video: {video_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio from {video_path}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg to process video files.")
        return False


def combine_audio_video(video_path, audio_path, output_path):
    """Combine processed audio with original video using ffmpeg"""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",  # Copy video stream as-is
            "-c:a",
            "aac",  # Encode audio as AAC
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",  # Map video from input 0, audio from input 1
            "-shortest",  # End when shortest stream ends
            "-y",  # Overwrite output
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Video created with enhanced audio: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to combine audio and video: {e.stderr}")
        return False


def process_audio_files(input_files, model, df_state, suffix, output_dir):
    """Process audio files with DeepFilterNet"""
    df_sr = ModelParams().sr
    ds = AudioDataset(input_files, df_sr)
    loader = DataLoader(ds, num_workers=2, pin_memory=True)
    n_samples = len(ds)

    processed_files = []

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
            f"{p_str}Enhanced audio file '{fn}' in {t:.2f}s (RT factor: {rtf:.3f})"
        )

        audio = resample(audio.to("cpu"), df_sr, audio_sr)

        # Generate output path manually since save_audio might not return it
        input_path = Path(file)
        output_filename = f"{input_path.stem}{suffix}{input_path.suffix}"
        output_path = os.path.join(output_dir, output_filename)

        # Save the audio file
        save_audio(
            file, audio, sr=audio_sr, output_dir=output_dir, suffix=suffix, log=False
        )

        # Verify the file was created and get the actual path
        if os.path.exists(output_path):
            processed_files.append(output_path)
        else:
            # Try to find the file with the suffix pattern
            possible_files = glob.glob(
                os.path.join(output_dir, f"{input_path.stem}*{input_path.suffix}")
            )
            if possible_files:
                output_path = possible_files[0]  # Take the first match
                processed_files.append(output_path)
            else:
                logger.error(f"Could not find processed audio file for {file}")
                continue

    return processed_files


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [output_file]")
        print("Supports both audio and video files")
        print("Video files require ffmpeg to be installed")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = "./temp"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Initialize DeepFilterNet model
    logger.info("Initializing DeepFilterNet model...")
    model, df_state, suffix = init_df()

    # Check if input is video or audio
    if is_video_file(input_file):
        logger.info(f"Processing video file: {input_file}")

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio from video
            temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")

            if not extract_audio_from_video(input_file, temp_audio_path):
                logger.error("Failed to extract audio from video")
                sys.exit(1)

            # Process the extracted audio
            logger.info("Processing extracted audio...")
            processed_files = process_audio_files(
                [temp_audio_path], model, df_state, suffix, temp_dir
            )

            if not processed_files:
                logger.error("Failed to process audio")
                sys.exit(1)

            processed_audio_path = processed_files[0]

            # Determine output path for video
            if output_file:
                final_output_path = output_file
            else:
                # Create output filename with suffix
                input_path = Path(input_file)
                final_output_path = os.path.join(
                    output_dir, f"{input_path.stem}{suffix}{input_path.suffix}"
                )

            # Combine processed audio with original video
            logger.info("Combining processed audio with video...")
            if combine_audio_video(input_file, processed_audio_path, final_output_path):
                logger.info(f"Video processing complete! Output: {final_output_path}")
            else:
                logger.error("Failed to create final video")
                sys.exit(1)

    else:
        # Process as audio file (original functionality)
        logger.info(f"Processing audio file: {input_file}")
        processed_files = process_audio_files(
            [input_file], model, df_state, suffix, output_dir
        )

        if processed_files:
            logger.info(f"Audio processing complete! Output: {processed_files[0]}")

            # If output_file specified, move/rename the processed file
            if output_file:
                import shutil

                shutil.move(processed_files[0], output_file)
                logger.info(f"File moved to: {output_file}")
        else:
            logger.error("Failed to process audio file")
            sys.exit(1)


if __name__ == "__main__":
    main()
