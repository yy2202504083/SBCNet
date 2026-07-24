"""Generate SBCNet prediction maps on COD test datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from models.SBCNet import SBCNet


# ImageNet normalization used by the PVTv2 backbone.
IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)

SUPPORTED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    """Load an SBCNet checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model",
            "model_state_dict",
        ):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint does not contain a valid state_dict."
        )

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys:
        print(
            f"[Warning] Missing keys: {len(missing_keys)}"
        )

    if unexpected_keys:
        print(
            f"[Warning] Unexpected keys: {len(unexpected_keys)}"
        )


def preprocess_image(
    image: np.ndarray,
    image_size: int,
) -> torch.Tensor:
    """Resize and normalize one RGB image."""
    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    resized = resized.astype(np.float32) / 255.0
    resized = (resized - IMAGE_MEAN) / IMAGE_STD

    tensor = torch.from_numpy(
        resized.transpose(2, 0, 1)
    ).unsqueeze(0)

    return tensor


def normalize_prediction(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Convert model output to a probability map."""
    prediction = prediction.float()

    if prediction.min() < 0.0 or prediction.max() > 1.0:
        prediction = torch.sigmoid(prediction)

    return prediction.clamp(0.0, 1.0)


def save_prediction(
    prediction: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a probability map as an 8-bit PNG image."""
    prediction_array = (
        prediction.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    prediction_array = (
        prediction_array * 255.0
    ).round().astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        prediction_array,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save prediction: {output_path}"
        )


def get_test_images(
    image_dir: Path,
) -> list[Path]:
    """Return all valid test images in one dataset."""
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Test image directory not found: {image_dir}"
        )

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


@torch.inference_mode()
def predict_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    test_root: Path,
    output_root: Path,
    image_size: int,
    device: torch.device,
) -> None:
    """Generate prediction maps for one test dataset."""
    image_dir = (
        test_root
        / dataset_name
        / "Imgs"
    )

    output_dir = (
        output_root
        / dataset_name
    )

    image_paths = get_test_images(
        image_dir
    )

    if not image_paths:
        print(
            f"[Warning] No images found in: {image_dir}"
        )
        return

    print(
        f"[{dataset_name}] "
        f"Testing {len(image_paths)} images."
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(
                f"[Warning] Failed to read: {image_path}"
            )
            continue

        original_height, original_width = image.shape[:2]

        input_tensor = preprocess_image(
            image=image,
            image_size=image_size,
        ).to(
            device,
            non_blocking=True,
        )

        prediction = model(
            input_tensor
        )

        # Compatible with models that still return training-style outputs.
        if isinstance(prediction, tuple):
            _, region_predictions = prediction
            prediction = region_predictions[-1]

        if isinstance(prediction, list):
            prediction = prediction[-1]

        prediction = normalize_prediction(
            prediction
        )

        prediction = F.interpolate(
            prediction,
            size=(
                original_height,
                original_width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        output_path = (
            output_dir
            / f"{image_path.stem}.png"
        )

        save_prediction(
            prediction,
            output_path,
        )

        print(
            f"\r[{dataset_name}] "
            f"{index}/{len(image_paths)}",
            end="",
            flush=True,
        )

    print(
        f"\n[{dataset_name}] "
        f"Predictions saved to: {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test SBCNet and save prediction maps."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the SBCNet model configuration.",
    )

    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to the trained SBCNet checkpoint.",
    )

    parser.add_argument(
        "--test-root",
        type=str,
        default="COD_Dataset/TestDataset",
        help="Root directory containing test datasets.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Directory used to save prediction maps.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=352,
        help="Model input resolution.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "CAMO",
            "CHAMELEON",
            "COD10K",
            "NC4K",
        ],
        help="Names of the test datasets.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device, such as cuda, cuda:0, or cpu.",
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[Warning] CUDA is unavailable; using CPU."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    config = load_config(
        args.config
    )

    # Full trained weights are loaded below, so backbone pre-training
    # does not need to be loaded separately.
    model = SBCNet(
        config,
        pretrained=False,
    ).to(device)

    load_checkpoint(
        model=model,
        checkpoint_path=Path(args.weight),
    )

    model.eval()

    test_root = Path(
        args.test_root
    )

    output_root = Path(
        args.output
    )

    print(f"Device: {device}")
    print(f"Input size: {args.image_size}×{args.image_size}")
    print(f"Checkpoint: {args.weight}")

    for dataset_name in args.datasets:
        dataset_dir = (
            test_root
            / dataset_name
        )

        if not dataset_dir.exists():
            print(
                f"[Warning] Dataset not found, skipped: "
                f"{dataset_dir}"
            )
            continue

        predict_dataset(
            model=model,
            dataset_name=dataset_name,
            test_root=test_root,
            output_root=output_root,
            image_size=args.image_size,
            device=device,
        )


if __name__ == "__main__":
    main()
