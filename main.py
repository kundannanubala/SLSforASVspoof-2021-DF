import argparse
import os
import sys

from utils.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SSL_PATH,
)
from utils.dataset import collect_audio_files
from utils.device import resolve_device


def validate_paths(input_dir, model_path, ssl_path):
    errors = []
    if not os.path.isdir(input_dir):
        errors.append(f"Input directory not found: {input_dir}")
    elif not collect_audio_files(input_dir):
        errors.append(f"No audio files found in {input_dir}")
    if not os.path.isfile(model_path):
        errors.append(f"Model checkpoint not found: {model_path}")
    if not os.path.isfile(ssl_path):
        errors.append(f"XLS-R backbone not found: {ssl_path}")
    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deepfake detection inference on audio files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing input audio files (recursive scan)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--ssl_path",
        type=str,
        default=DEFAULT_SSL_PATH,
        help="Path to XLS-R 300M backbone checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write scores and results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional score threshold for bonafide/spoof labels",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        default=False,
        help="Skip writing results.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_paths(args.input_dir, args.model_path, args.ssl_path)

    from utils.loader import load_model
    from utils.predictor import InferenceEngine

    device = resolve_device(args.device)
    print(f"Device: {device}")

    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.ssl_path, device)

    engine = InferenceEngine(model, device)
    engine.predict_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        write_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
