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


def center_distance_score(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    avg_size = max(1, ((aw + ah + bw + bh) / 4))
    dist = np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)

    return dist / avg_size


def smooth_box(old_box, new_box, alpha=0.70):
    return [
        int(old_box[0] * (1 - alpha) + new_box[0] * alpha),
        int(old_box[1] * (1 - alpha) + new_box[1] * alpha),
        int(old_box[2] * (1 - alpha) + new_box[2] * alpha),
        int(old_box[3] * (1 - alpha) + new_box[3] * alpha),
    ]


def scale_box_xy(box, x_scale, y_scale, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    nw = bw * x_scale
    nh = bh * y_scale

    nx1 = int(max(0, cx - nw / 2))
    ny1 = int(max(0, cy - nh / 2))
    nx2 = int(min(frame_w, cx + nw / 2))
    ny2 = int(min(frame_h, cy + nh / 2))

    return nx1, ny1, nx2, ny2


def scale_box(box, scale, frame_w, frame_h):
    return scale_box_xy(box, scale, scale, frame_w, frame_h)


def valid_face_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return False

    # Ignore tiny noisy detections.
    min_face_size = max(16, int(min(frame_w, frame_h) * 0.012))
    if bw < min_face_size or bh < min_face_size:
        return False

    # Ignore extreme shapes, but keep this loose for side/profile faces.
    aspect = bw / float(bh)
    if aspect < 0.30 or aspect > 2.60:
        return False

    # Ignore huge false detections.
    frame_area = frame_w * frame_h
    box_area = bw * bh
    if box_area > frame_area * 0.45:
        return False

    return True


def apply_dark_oval_blur(frame, box, box_scale=1.25, darkness=1.0):
    frame_h, frame_w = frame.shape[:2]

    # Original oval behavior.
    x1, y1, x2, y2 = scale_box(box, box_scale, frame_w, frame_h)

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # Clear blur only. No black/dark tint.
    blurred_face = cv2.GaussianBlur(roi, (151, 151), 80)

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

    # Original oval shape.
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

    # Soft oval edge.
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = mask[..., None] / 255.0

    # Natural color blur. No darkness multiplier.
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
    parser.add_argument("--pixel-size", type=int, default=30)
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
    print("Rule: blur every detected face with temporal memory, smoothing, and side-face padding.")
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

            det = [x1, y1, x2, y2]

            if valid_face_box(det, w, h):
                detections.append(det)

        # Ignore the main foreground/task person.
        # He is usually closest to camera, so his face box is usually the largest.
        # Background faces are usually smaller and will still be blurred.
        if detections:
            frame_area = w * h

            def area(det):
                return max(0, det[2] - det[0]) * max(0, det[3] - det[1])

            largest_det = max(detections, key=area)
            largest_area = area(largest_det)

            # Only ignore it if it is clearly a foreground-sized face.
            # If this is too aggressive, increase 0.006 to 0.010.
            # If it still blurs the main person, decrease 0.006 to 0.003.
            foreground_area_thresh = frame_area * 0.003

            detections = [
                det for det in detections
                if not (det == largest_det and largest_area >= foreground_area_thresh)
            ]

        used_dets = set()

        for track in tracks:
            best_score = -999
            best_j = None

            for j, det in enumerate(detections):
                if j in used_dets:
                    continue

                iou_score = iou(track["box"], det)
                dist_score = center_distance_score(track["box"], det)

                # Match if IoU is good OR centers are close.
                # This helps when side/profile boxes shift shape.
                combined = iou_score - (dist_score * 0.15)

                if combined > best_score:
                    best_score = combined
                    best_j = j

            if best_j is not None:
                det = detections[best_j]
                iou_score = iou(track["box"], det)
                dist_score = center_distance_score(track["box"], det)

                if iou_score >= args.match_iou or dist_score <= 0.90:
                    track["box"] = smooth_box(track["box"], det, alpha=0.70)
                    track["missed"] = 0
                    used_dets.add(best_j)
                else:
                    track["missed"] += 1
            else:
                track["missed"] += 1

        for j, det in enumerate(detections):
            if j not in used_dets:
                tracks.append({"box": det, "missed": 0})

        tracks = [t for t in tracks if t["missed"] <= args.memory_frames]

        output_frame = frame.copy()

        for track in tracks:
            # When detector briefly loses a side face, slightly expand remembered blur.
            missed_ratio = 0
            if args.memory_frames > 0:
                missed_ratio = min(1.0, track["missed"] / float(args.memory_frames))

            adaptive_scale = args.box_scale * (1.0 + 0.22 * missed_ratio)

            apply_dark_oval_blur(
                output_frame,
                track["box"],
                box_scale=adaptive_scale,
                darkness=args.darkness
            )

        writer.write(output_frame)

        if debug_writer is not None:
            debug_frame = output_frame.copy()

            for track in tracks:
                missed_ratio = 0
                if args.memory_frames > 0:
                    missed_ratio = min(1.0, track["missed"] / float(args.memory_frames))

                adaptive_scale = args.box_scale * (1.0 + 0.22 * missed_ratio)

                x1, y1, x2, y2 = scale_box_xy(
                    track["box"],
                    adaptive_scale * 1.22,
                    adaptive_scale * 1.08,
                    w,
                    h
                )

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
