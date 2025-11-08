#!/usr/bin/env python3
"""Gradio GUI for YOLOv11 + SAHI inference (original simple style).

Features:
 - Load Ultralytics model (.pt)
 - Sliced inference with SAHI
 - Colored bounding boxes per class
 - COCO-format JSON export
 - Cropped detections (224x224, padded square) ZIP export
 - Original filename preservation
"""

import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Union

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Color palette for classes (repeat as needed)
PALETTE = [
    (220, 20, 60),
    (34, 139, 34),
    (30, 144, 255),
    (255, 140, 0),
    (138, 43, 226),
    (0, 206, 209),
    (199, 21, 133),
    (255, 215, 0),
    (70, 130, 180),
    (154, 205, 50),
    (255, 99, 71),
    (0, 191, 255),
    (218, 112, 214),
    (244, 164, 96),
    (46, 139, 87),
]


def _infer_image_filename(image_input: Union[str, Image.Image]) -> str:
    if isinstance(image_input, str):
        return Path(image_input).name
    return "uploaded_image.jpg"


def draw_predictions(image_pil: Image.Image, object_prediction_list) -> Image.Image:
    vis = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for idx, obj in enumerate(object_prediction_list):
        x1, y1, x2, y2 = obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy
        cls_name = obj.category.name
        score = obj.score.value
        color = PALETTE[obj.category.id % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{cls_name} {score:.2f}"
        tw, th = draw.textsize(text, font=font)
        pad = 2
        draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=color)
        draw.text((x1 + pad, y1 - th - pad), text, fill="white", font=font)
    return vis


def run_inference(
    model_type: str,
    model_path: str,
    device: str,
    conf_thres: float,
    image_size: int,
    image_input: Union[str, Image.Image],
    slice_height: int,
    slice_width: int,
    overlap_h: float,
    overlap_w: float,
    batch_size: int,
    verbose: int,
):
    # Load image
    if isinstance(image_input, str):
        image_pil = Image.open(image_input).convert("RGB")
        orig_filename = _infer_image_filename(image_input)
    else:
        image_pil = image_input.convert("RGB")
        orig_filename = _infer_image_filename(image_input)

    # Load model via SAHI AutoDetectionModel
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf_thres,
        device=device,
        image_size=image_size,
    )

    result = get_sliced_prediction(
        image=image_pil,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        batch_size=batch_size,
        verbose=verbose,
    )

    obj_list = result.object_prediction_list
    vis = draw_predictions(image_pil, obj_list)

    # Build COCO JSON
    images = [
        {
            "id": 1,
            "file_name": orig_filename,
            "width": image_pil.width,
            "height": image_pil.height,
        }
    ]
    categories = {}
    annotations = []
    ann_id = 1
    for obj in obj_list:
        cid = obj.category.id
        if cid not in categories:
            categories[cid] = {"id": cid, "name": obj.category.name}
        x1, y1, x2, y2 = obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy
        w = x2 - x1
        h = y2 - y1
        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": cid,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "score": obj.score.value,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": list(categories.values()),
    }

    json_fd, json_path = tempfile.mkstemp(prefix="pred_", suffix=".json")
    os.close(json_fd)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    # Crops with padded square context
    TARGET_SIZE = 224
    crops: List[Tuple[Image.Image, str]] = []
    zip_fd, zip_path = tempfile.mkstemp(prefix="crops_", suffix=".zip")
    os.close(zip_fd)
    zf = zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED)
    used_names = set()
    base_stem = Path(orig_filename).stem

    for obj in obj_list:
        x1, y1, x2, y2 = obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy
        # add 25% padding each side
        pad_x = 0.25 * (x2 - x1)
        pad_y = 0.25 * (y2 - y1)
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(image_pil.width, int(x2 + pad_x))
        cy2 = min(image_pil.height, int(y2 + pad_y))
        # make square by expanding smaller dimension
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        side = max(crop_w, crop_h)
        extra_w = side - crop_w
        extra_h = side - crop_h
        cx1 = max(0, cx1 - extra_w // 2)
        cx2 = min(image_pil.width, cx2 + math.ceil(extra_w / 2))
        cy1 = max(0, cy1 - extra_h // 2)
        cy2 = min(image_pil.height, cy2 + math.ceil(extra_h / 2))
        crop_square = image_pil.crop((cx1, cy1, cx2, cy2))
        crop_resized = crop_square.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
        label = f"{obj.category.name} {obj.score.value:.2f}"
        crops.append((crop_resized, label))
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="crop_", suffix=".jpg")
        os.close(tmp_fd)
        crop_resized.save(tmp_path, format="JPEG", quality=90)
        cls_name = str(obj.category.name).strip().replace(" ", "_")
        conf_txt = f"{obj.score.value:.2f}"
        base_name = f"{base_stem}_{cls_name}_{conf_txt}.jpg"
        final_name = base_name
        k = 2
        while final_name in used_names:
            final_name = f"{base_stem}_{cls_name}_{conf_txt}_{k}.jpg"
            k += 1
        used_names.add(final_name)
        zf.write(tmp_path, arcname=final_name)
        os.remove(tmp_path)
    zf.close()

    summary = f"Detections: {len(obj_list)} | JSON: {Path(json_path).name} | Crops: {len(crops)}"
    return vis, summary, json_path, crops, zip_path


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YOLOv11 SAHI Inference") as demo:
        gr.Markdown(
            """
        # YOLOv11 SAHI Inference
        Load a YOLOv11/Ultralytics model and run sliced inference with SAHI. Visualize predictions on your image.
        """
        )

        model_type = gr.Dropdown(
            label="Model Type",
            choices=["ultralytics"],
            value="ultralytics",
        )
        model_path = gr.File(
            label="Model File (.pt)",
            file_types=[".pt", ".pth", ".onnx"],
            type="filepath",
        )
        device = gr.Dropdown(
            label="Device",
            choices=["mps", "cpu", "cuda"],
            value="mps",
            info="MPS: Apple Silicon (M1/M2/M3), CUDA: NVIDIA GPU (Windows/Linux), CPU: All platforms (slower)",
        )
        conf_thres = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence Threshold")
        image_size = gr.Slider(256, 1280, value=640, step=32, label="Inference Image Size", info="Ultralytics imgsz")

        gr.Markdown("### SAHI Slice Settings")
        slice_height = gr.Slider(128, 2048, value=640, step=32, label="Slice Height")
        slice_width = gr.Slider(128, 2048, value=640, step=32, label="Slice Width")
        overlap_h = gr.Slider(0.0, 0.9, value=0.3, step=0.05, label="Overlap Height Ratio")
        overlap_w = gr.Slider(0.0, 0.9, value=0.3, step=0.05, label="Overlap Width Ratio")
        batch_size = gr.Slider(1, 64, value=10, step=1, label="Batch Size")
        verbose = gr.Dropdown([0, 1, 2], value=0, label="Verbose")
        run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Input Image", sources=["upload"], image_mode="RGB")
            output_image_pred = gr.Image(type="pil", label="Predictions")

        with gr.Row():
            output_text = gr.Textbox(label="Summary", lines=4)
            output_json = gr.File(label="Predictions JSON")
            output_zip = gr.File(label="Crops ZIP")

        output_gallery = gr.Gallery(label="Detections (crops)", columns=4, preview=True)

        run_btn.click(
            fn=run_inference,
            inputs=[
                model_type,
                model_path,
                device,
                conf_thres,
                image_size,
                image_input,
                slice_height,
                slice_width,
                overlap_h,
                overlap_w,
                batch_size,
                verbose,
            ],
            outputs=[output_image_pred, output_text, output_json, output_gallery, output_zip],
        )

        gr.Markdown(
            """
        Tips:
        - CPU forces sequential processing (batch_size=1 internally) for best speed.
        - Increase batch size only when using GPU/MPS and memory allows.
        - Larger slice sizes reduce number of slices but may miss small objects; tune per use case.
        """
        )
    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, max_file_size="50mb")
