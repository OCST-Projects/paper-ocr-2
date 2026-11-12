"""Run an OCST-MAIE enhancement checkpoint on a single image."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import MambaRestorationNet, ResidualDiffusionRefiner, UNetRestorer, sobel_edges


MODEL_CHECKPOINTS = {
    "nafnet_lite": Path("weights/nafnet_lite_best.pt"),
    "restormer_lite": Path("weights/restormer_lite_best.pt"),
    "swinir_lite": Path("weights/swinir_lite_best.pt"),
    "mamba": Path("weights/mamba_raw_to_sam3_best.pt"),
    "residual": Path("weights/residual_refiner_best.pt"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = int(config.get("base_channels", 32))
    if checkpoint.get("model_type") == "mamba_like_restoration":
        model = MambaRestorationNet(base_channels=base_channels).to(device)
    elif "condition_channels" in checkpoint:
        model = ResidualDiffusionRefiner(
            in_channels=int(checkpoint["condition_channels"]), base_channels=base_channels
        ).to(device)
    else:
        family = checkpoint.get("family")
        if family not in {"naf", "restormer", "swinir"}:
            raise ValueError(f"Unsupported checkpoint family: {family}")
        model = UNetRestorer(base_channels=base_channels, family=str(family)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(config.get("image_size", 256))


def load_naf_base_model(device: torch.device) -> UNetRestorer:
    """Load the deterministic Plan 1 base model required by Plan 4."""
    checkpoint = torch.load(MODEL_CHECKPOINTS["nafnet_lite"], map_location=device, weights_only=False)
    model = UNetRestorer(
        base_channels=int(checkpoint.get("config", {}).get("base_channels", 32)),
        family="naf",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_input(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor, (height, width)


def save_output(prediction: torch.Tensor, output_path: Path, original_size: tuple[int, int]) -> None:
    rgb = (
        prediction.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5
    ).astype(np.uint8)
    height, width = original_size
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write output image: {output_path}")


def enhance_image(
    image_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    base_model: UNetRestorer | None = None,
) -> None:
    """Enhance one image with a preloaded model and preserve its original dimensions."""
    image, original_size = load_input(image_path, image_size)
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        if base_model is not None:
            base = base_model(image).clamp(0, 1)
            edge = sobel_edges(image).mean(dim=1, keepdim=True).clamp(0, 1)
            condition = torch.cat([image, base, torch.zeros_like(base), edge], dim=1)
            time = torch.zeros(image.shape[0], device=device, dtype=torch.long)
            prediction = (base + model(condition, time) * 0.25).clamp(0, 1)[0]
        else:
            prediction = model(image)[0]
    save_output(prediction, output_path, original_size)


def image_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance one image or a folder with OCST-MAIE.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="Path to one input image.")
    inputs.add_argument("--input-dir", type=Path, help="Directory of input images.")
    parser.add_argument("--model", choices=MODEL_CHECKPOINTS, default="nafnet_lite")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional custom checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output path for --input mode.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --input-dir mode.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is None and args.input_dir is None:
        args.input_dir = Path("test_images")
    if args.input is not None and args.output_dir is not None:
        raise ValueError("--output-dir can only be used with --input-dir")
    if args.input_dir is not None and args.output is not None:
        raise ValueError("--output can only be used with --input")

    device = select_device(args.device)
    checkpoint_path = args.checkpoint or MODEL_CHECKPOINTS[args.model]
    model, image_size = load_model(checkpoint_path, device)
    base_model = load_naf_base_model(device) if args.model == "residual" else None

    print(f"Device: {device}")
    print(f"Inference size: {image_size}x{image_size}")
    if args.input is not None:
        output_path = args.output or Path("outputs") / f"{args.input.stem}_{args.model}.jpg"
        enhance_image(args.input, output_path, model, image_size, device, base_model)
        print(f"Saved result: {output_path}")
        return

    output_dir = args.output_dir or Path("outputs") / args.model
    paths = image_files(args.input_dir)
    if not paths:
        raise RuntimeError(f"No supported images found in: {args.input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(paths, start=1):
        output_path = output_dir / image_path.name
        enhance_image(image_path, output_path, model, image_size, device, base_model)
        print(f"[{index}/{len(paths)}] Saved result: {output_path}")


if __name__ == "__main__":
    main()
