print("--- VERIFICATION: FISH DETECTION VERSION LOADING ---")

import os
import sys
import csv
import cv2
import torch
import numpy as np
import argparse
import time
import math
from datetime import datetime
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from skimage import color as skcolor

# =========================
# FUnIE-GAN IMPORT
# =========================
FUNIEGAN_DIR = os.path.expanduser("~/Downloads/FUnIE-GAN-master/PyTorch")
sys.path.insert(0, FUNIEGAN_DIR)

try:
    from nets.funiegan import GeneratorFunieGAN
    print("FUnIE-GAN imported successfully.")
except ImportError as e:
    print(f"Error importing FUnIE-GAN: {e}")
    sys.exit(1)

# =========================
# ARGS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, required=True)

# FUnIE-GAN
parser.add_argument(
    "--model_path",
    type=str,
    default=os.path.join(
        FUNIEGAN_DIR,
        "pretrained_models/funie_generator.pth"
    )
)

# YOUR FISH MODEL
parser.add_argument(
    "--fish_model",
    type=str,
    default="f4k_single_m.pt"
)

# Detection settings
parser.add_argument("--conf", type=float, default=0.20)
parser.add_argument("--iou", type=float, default=0.40)

# Enhancement settings
parser.add_argument("--img_size", type=int, default=256)

# YOLO inference size
parser.add_argument("--yolo_size", type=int, default=960)

# Skip detection every N frames
parser.add_argument("--detect_every", type=int, default=2)

# Metrics interval
parser.add_argument("--metric_every", type=int, default=10)

# SAHI slicing
parser.add_argument("--sahi_slice", type=int, default=2)
parser.add_argument("--sahi_overlap", type=float, default=0.20)

# Display
parser.add_argument("--no_display", action="store_true")

opt = parser.parse_args()

# =========================
# OUTPUT DIRECTORY
# =========================
input_dir = os.path.dirname(os.path.abspath(opt.input))
input_stem = os.path.splitext(os.path.basename(opt.input))[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_name = f"{input_stem}_{timestamp}"

out_dir = os.path.join(input_dir, run_name)
os.makedirs(out_dir, exist_ok=True)

output_video = os.path.join(
    out_dir,
    f"{run_name}_fish_detection.mp4"
)

output_csv = os.path.join(
    out_dir,
    f"{run_name}_metrics.csv"
)

print("\n" + "=" * 60)
print(f"Output folder : {out_dir}")
print(f"Output video  : {output_video}")
print(f"Metrics CSV   : {output_csv}")
print("=" * 60 + "\n")

# =========================
# DEVICE
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"

print(f"Using device: {device}")
print(f"AMP enabled : {use_amp}")

# =========================
# LOAD FUNIE-GAN
# =========================
funiegan = GeneratorFunieGAN().to(device)

funiegan.load_state_dict(
    torch.load(
        opt.model_path,
        map_location=device,
        weights_only=False
    )
)

funiegan.eval()

print(f"FUnIE-GAN loaded : {opt.model_path}")

# =========================
# LOAD FISH MODEL
# =========================
fish_model = YOLO(opt.fish_model).to(device)

print(f"Fish detector loaded : {opt.fish_model}")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =========================
# VIDEO I/O
# =========================
cap = cv2.VideoCapture(opt.input)

if not cap.isOpened():
    print(f"Cannot open video: {opt.input}")
    sys.exit(1)

src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (src_w * 2, src_h)
)

print(
    f"Input : {opt.input} "
    f"({src_w}x{src_h} @ {fps:.1f} FPS)"
)

# =========================
# METRICS
# =========================
def compute_uciqe(bgr):

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    ) / 255.0

    lab = skcolor.rgb2lab(rgb).astype(np.float32)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    C = np.sqrt(a**2 + b**2)

    sigma_c = float(np.std(C))

    l_flat = L.flatten()

    k = max(1, int(0.01 * len(l_flat)))

    l_sort = np.sort(l_flat)

    con_L = float(
        np.mean(l_sort[-k:]) -
        np.mean(l_sort[:k])
    )

    denom = np.sqrt(L**2 + C**2)

    s = np.divide(
        C,
        denom,
        out=np.zeros_like(C),
        where=denom > 1e-7
    )

    return round(
        0.4680 * sigma_c +
        0.2745 * con_L +
        0.2576 * float(np.mean(s)),
        4
    )

def _eme(channel_uint8, block_size=8):

    h, w = channel_uint8.shape

    score = 0.0
    count = 0

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):

            blk = channel_uint8[
                y:y+block_size,
                x:x+block_size
            ].astype(np.float64)

            i_max = float(blk.max())
            i_min = max(float(blk.min()), 1.0)

            if i_max >= 1.0:
                score += math.log(i_max / i_min)

            count += 1

    return (2.0 / count) * score if count > 0 else 0.0

def _amee(gray_uint8, block_size=8):

    h, w = gray_uint8.shape

    score = 0.0
    count = 0

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):

            blk = gray_uint8[
                y:y+block_size,
                x:x+block_size
            ].astype(np.float64)

            i_max = float(blk.max())
            i_min = float(blk.min())

            if (i_max + i_min) > 0.0 and i_max > i_min:

                m = (
                    (i_max - i_min) /
                    (i_max + i_min)
                )

                score += m * math.log(m)

            count += 1

    return -(score / count) if count > 0 else 0.0

def compute_uiqm(bgr):

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    R8 = rgb[:, :, 0]
    G8 = rgb[:, :, 1]
    B8 = rgb[:, :, 2]

    Rf = R8 / 255.0
    Gf = G8 / 255.0
    Bf = B8 / 255.0

    RG = Rf - Gf
    YB = (Rf + Gf) / 2.0 - Bf

    uicm = (
        -0.0268 *
        math.sqrt(np.mean(RG)**2 + np.mean(YB)**2)
        +
        0.1586 *
        math.sqrt(np.std(RG)**2 + np.std(YB)**2)
    )

    uism = (
        0.299 * _eme(R8) +
        0.587 * _eme(G8) +
        0.114 * _eme(B8)
    )

    uiconm = _amee(
        cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )
    )

    return round(
        float(
            0.0282 * uicm +
            0.2953 * uism +
            3.5753 * uiconm
        ),
        4
    )

def compute_niqe(bgr):

    try:
        import piq

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        t = torch.from_numpy(rgb)\
            .permute(2, 0, 1)\
            .unsqueeze(0)\
            .float()\
            .to(device) / 255.0

        score = piq.niqe(
            t,
            data_range=1.0
        )

        return round(float(score.item()), 4)

    except Exception:

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float64)

        sigma = np.std(gray)

        if sigma > 0:

            kurt = np.mean(
                ((gray - np.mean(gray)) / sigma) ** 4
            )

            return round(
                abs(float(kurt) - 3.0) + 2.0,
                4
            )

        return 0.0

# =========================
# ENHANCEMENT
# =========================
def enhance_frame(bgr):

    pil_img = Image.fromarray(
        cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )
    )

    t = transform(pil_img)\
        .unsqueeze(0)\
        .to(device)

    with torch.no_grad():
        out = funiegan(t)

    out = (
        out.squeeze(0).cpu() * 0.5 + 0.5
    ).clamp(0, 1)

    enh = cv2.cvtColor(
        np.array(
            transforms.ToPILImage()(out)
        ),
        cv2.COLOR_RGB2BGR
    )

    return cv2.resize(
        enh,
        (src_w, src_h)
    )

# =========================
# TEXT
# =========================
def overlay_text(
    frame,
    text,
    pos,
    color=(0, 255, 180),
    scale=0.65
):

    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA
    )

# =========================
# DRAW BOXES
# =========================
def draw_boxes(frame, boxes):

    for (
        x1,
        y1,
        x2,
        y2,
        conf_score,
        label
    ) in boxes:

        color = (0, 255, 180)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        tag = f"FISH {conf_score:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            tag,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        lx1 = x1
        ly1 = max(
            y1 - th - baseline - 6,
            0
        )

        lx2 = x1 + tw + 10
        ly2 = y1

        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx2, ly2),
            color,
            -1
        )

        cv2.putText(
            frame,
            tag,
            (lx1 + 4, ly2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return frame

# =========================
# METRIC PANEL
# =========================
def draw_metric_panel(
    frame,
    metrics,
    num_det,
    fps_val
):

    panel_h = 130

    h = frame.shape[0]
    w = frame.shape[1]

    bg = frame.copy()

    cv2.rectangle(
        bg,
        (0, h - panel_h),
        (w, h),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        bg,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    uciqe, uiqm, niqe = metrics

    lines = [
        (
            f"UCIQE : {uciqe:.4f}",
            (0, 255, 180)
        ),
        (
            f"UIQM  : {uiqm:.4f}",
            (0, 220, 255)
        ),
        (
            f"NIQE  : {niqe:.4f}",
            (255, 200, 0)
        ),
        (
            f"Fish Detected : {num_det} | FPS : {fps_val:.1f}",
            (200, 200, 200)
        )
    ]

    for i, (txt, col) in enumerate(lines):

        y = h - panel_h + 26 + i * 26

        cv2.putText(
            frame,
            txt,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            txt,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
            cv2.LINE_AA
        )

    return frame

# =========================
# CSV LOG
# =========================
csv_file = open(output_csv, "w", newline="")

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "timestamp_s",
    "uciqe",
    "uiqm",
    "niqe",
    "fish_count",
    "fps"
])

# =========================
# MAIN LOOP
# =========================
frame_idx = 0

fps_smooth = 0.0

t_start = time.time()

boxes_cache = []

metrics = (0.0, 0.0, 0.0)

# =========================
# DISPLAY
# =========================
show_display = not opt.no_display

if show_display:

    cv2.namedWindow(
        "Underwater Fish Detection",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Underwater Fish Detection",
        min(src_w * 2, 1400),
        min(src_h, 700)
    )

    print("Live preview : ON")

else:
    print("Live preview : OFF")

print("\nProcessing started...\n")

# =========================
# LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    t0 = time.time()

    # =========================
    # ENHANCE
    # =========================
    enhanced = enhance_frame(frame)

    # =========================
    # DETECT
    # =========================
    if frame_idx % opt.detect_every == 0:

        raw_boxes = []

        n = opt.sahi_slice
        overlap = opt.sahi_overlap

        H, W = enhanced.shape[:2]

        step_x = int(W / n)
        step_y = int(H / n)

        pad_x = int(step_x * overlap)
        pad_y = int(step_y * overlap)

        # =========================
        # SAHI PATCH DETECTION
        # =========================
        for row in range(n):

            for col in range(n):

                px1 = max(col * step_x - pad_x, 0)
                py1 = max(row * step_y - pad_y, 0)

                px2 = min(
                    (col + 1) * step_x + pad_x,
                    W
                )

                py2 = min(
                    (row + 1) * step_y + pad_y,
                    H
                )

                patch = enhanced[
                    py1:py2,
                    px1:px2
                ]

                results = fish_model(
                    patch,
                    imgsz=opt.yolo_size,
                    conf=opt.conf,
                    iou=opt.iou,
                    verbose=False
                )

                for r in results:

                    for box in r.boxes:

                        bx1, by1, bx2, by2 = map(
                            int,
                            box.xyxy[0]
                        )

                        conf_score = float(
                            box.conf[0]
                        )

                        fx1 = bx1 + px1
                        fy1 = by1 + py1

                        fx2 = bx2 + px1
                        fy2 = by2 + py1

                        raw_boxes.append(
                            (
                                fx1,
                                fy1,
                                fx2,
                                fy2,
                                conf_score
                            )
                        )

        # =========================
        # FULL FRAME DETECTION
        # =========================
        results = fish_model(
            enhanced,
            imgsz=opt.yolo_size,
            conf=opt.conf,
            iou=opt.iou,
            verbose=False
        )

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf_score = float(
                    box.conf[0]
                )

                raw_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        conf_score
                    )
                )

        # =========================
        # REMOVE DUPLICATES
        # =========================
        boxes_cache = []

        used = [False] * len(raw_boxes)

        raw_boxes.sort(
            key=lambda b: b[4],
            reverse=True
        )

        for i, (
            x1,
            y1,
            x2,
            y2,
            cs
        ) in enumerate(raw_boxes):

            if used[i]:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            boxes_cache.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    cs,
                    "Fish"
                )
            )

            for j in range(i + 1, len(raw_boxes)):

                if used[j]:
                    continue

                ox1, oy1, ox2, oy2, _ = raw_boxes[j]

                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2

                if (
                    abs(cx - ocx) < 40 and
                    abs(cy - ocy) < 40
                ):
                    used[j] = True

    # =========================
    # METRICS
    # =========================
    if frame_idx % opt.metric_every == 0:

        metrics = (
            compute_uciqe(enhanced),
            compute_uiqm(enhanced),
            compute_niqe(enhanced)
        )

    # =========================
    # DRAW
    # =========================
    result_frame = enhanced.copy()

    result_frame = draw_boxes(
        result_frame,
        boxes_cache
    )

    elapsed = time.time() - t0

    fps_smooth = (
        0.9 * fps_smooth +
        0.1 * (
            1.0 / max(elapsed, 1e-6)
        )
    )

    left = frame.copy()
    right = result_frame.copy()

    overlay_text(
        left,
        "ORIGINAL",
        (10, 30)
    )

    overlay_text(
        right,
        "ENHANCED + FISH DETECTION",
        (10, 30)
    )

    overlay_text(
        left,
        f"Frame {frame_idx}/{total}",
        (10, src_h - 20)
    )

    right = draw_metric_panel(
        right,
        metrics,
        len(boxes_cache),
        fps_smooth
    )

    combined = np.hstack(
        (left, right)
    )

    writer.write(combined)

    # =========================
    # CSV
    # =========================
    csv_writer.writerow([
        frame_idx,
        round(frame_idx / fps, 3),
        metrics[0],
        metrics[1],
        metrics[2],
        len(boxes_cache),
        round(fps_smooth, 2)
    ])

    # =========================
    # DISPLAY
    # =========================
    if show_display:

        preview = cv2.resize(
            combined,
            (
                min(src_w * 2, 1400),
                min(src_h, 700)
            )
        )

        cv2.imshow(
            "Underwater Fish Detection",
            preview
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):

            print("\nStopped by user.")
            break

    # =========================
    # CONSOLE LOG
    # =========================
    if frame_idx % 30 == 0 or frame_idx == 1:

        print(
            f"Frame {frame_idx:5d}/{total} | "
            f"FPS: {fps_smooth:5.1f} | "
            f"Fish: {len(boxes_cache)} | "
            f"UCIQE: {metrics[0]:.4f} | "
            f"UIQM: {metrics[1]:.4f} | "
            f"NIQE: {metrics[2]:.4f}"
        )

# =========================
# CLEANUP
# =========================
cap.release()

writer.release()

csv_file.close()

cv2.destroyAllWindows()

total_time = time.time() - t_start

print("\n" + "=" * 60)
print(
    f"Done! {frame_idx} frames "
    f"in {total_time:.1f}s"
)

print(
    f"Average FPS : "
    f"{frame_idx / total_time:.1f}"
)

print(f"Saved video : {output_video}")
print(f"Saved CSV   : {output_csv}")

print("=" * 60)
