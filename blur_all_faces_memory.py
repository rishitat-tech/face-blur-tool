import argparse
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0
    return inter / union


def scale_box(box, scale, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    nx1 = int(max(0, cx - w / 2))
    ny1 = int(max(0, cy - h / 2))
    nx2 = int(min(frame_w, cx + w / 2))
    ny2 = int(min(frame_h, cy + h / 2))

    return nx1, ny1, nx2, ny2


def apply_dark_oval_blur(frame, box, box_scale=1.25, darkness=0.65):
    frame_h, frame_w = frame.shape[:2]

    x1, y1, x2, y2 = scale_box(box, box_scale, frame_w, frame_h)

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]

    blurred_face = cv2.GaussianBlur(roi, (151, 151), 80)
    blurred_face = (blurred_face * darkness).astype(np.uint8)

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

    cv2.ellipse(
        mask,
        ((x2 - x1) // 2, (y2 - y1) // 2),
        (int((x2 - x1) * 0.43), int((y2 - y1) * 0.55)),
        0,
        0,
        360,
        255,
        -1
    )

    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = mask[..., None] / 255.0

    frame[y1:y2, x1:x2] = (blurred_face * alpha + roi * (1 - alpha)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-output", default=None)
    parser.add_argument("--det-thresh", type=float, default=0.20)
    parser.add_argument("--box-scale", type=float, default=1.25)
    parser.add_argument("--memory-frames", type=int, default=8)
    parser.add_argument("--darkness", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.20)
    parser.add_argument("--det-size", type=int, default=960)
    parser.add_argument("--pixel-size", type=int, default=30)  # kept only so old commands don't break
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {args.input}")
    print(f"Size: {w}x{h}, FPS: {fps}, frames: {frames}")
    print("Rule: blur every detected face with temporal memory to reduce flicker.")
    print(f"Memory frames: {args.memory_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    debug_writer = None
    if args.debug_output:
        debug_writer = cv2.VideoWriter(args.debug_output, fourcc, fps, (w, h))

    print("Loading face detector...")
    detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    detector.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)

    tracks = []

    for _ in tqdm(range(frames)):
        ok, frame = cap.read()
        if not ok:
            break

        faces = detector.get(frame)

        detections = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 > x1 and y2 > y1:
                detections.append([x1, y1, x2, y2])

        used_dets = set()

        for track in tracks:
            best_iou = 0
            best_j = None

            for j, det in enumerate(detections):
                if j in used_dets:
                    continue

                score = iou(track["box"], det)
                if score > best_iou:
                    best_iou = score
                    best_j = j

            if best_j is not None and best_iou >= args.match_iou:
                track["box"] = detections[best_j]
                track["missed"] = 0
                used_dets.add(best_j)
            else:
                track["missed"] += 1

        for j, det in enumerate(detections):
            if j not in used_dets:
                tracks.append({"box": det, "missed": 0})

        tracks = [t for t in tracks if t["missed"] <= args.memory_frames]

        output_frame = frame.copy()

        for track in tracks:
            apply_dark_oval_blur(
                output_frame,
                track["box"],
                box_scale=args.box_scale,
                darkness=args.darkness
            )

        writer.write(output_frame)

        if debug_writer is not None:
            debug_frame = output_frame.copy()
            for track in tracks:
                x1, y1, x2, y2 = scale_box(track["box"], args.box_scale, w, h)
                color = (0, 255, 0) if track["missed"] == 0 else (0, 165, 255)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    debug_frame,
                    f"missed={track['missed']}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            debug_writer.write(debug_frame)

    cap.release()
    writer.release()

    if debug_writer is not None:
        debug_writer.release()

    print("Done.")
    print(f"Blurred video: {args.output}")
    if args.debug_output:
        print(f"Debug video: {args.debug_output}")


if __name__ == "__main__":
    main()
