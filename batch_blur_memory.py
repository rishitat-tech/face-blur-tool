import argparse
import subprocess
from pathlib import Path
import csv
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--log-csv", default="batch_outputs/log.csv")
    parser.add_argument("--det-thresh", default="0.40")
    parser.add_argument("--det-size", default="800")
    parser.add_argument("--box-scale", default="1.20")
    parser.add_argument("--memory-frames", default="4")
    parser.add_argument("--darkness", default="0.65")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    log_csv = Path(args.log_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mkv")))
    print(f"Found {len(videos)} videos")

    results = []

    for i, video in enumerate(videos, start=1):
        output_path = output_dir / f"{video.stem}_blurred.mp4"
        debug_path = debug_dir / f"{video.stem}_debug.mp4" if debug_dir else None

        if args.skip_existing and output_path.exists():
            print(f"[{i}/{len(videos)}] skipped existing: {video}")
            results.append({"video": str(video), "output": str(output_path), "status": "skipped_existing", "seconds": 0})
            continue

        cmd = [
            "python", "blur_all_faces_memory.py",
            "--input", str(video),
            "--output", str(output_path),
            "--det-thresh", args.det_thresh,
            "--det-size", args.det_size,
            "--box-scale", args.box_scale,
            "--memory-frames", args.memory_frames,
            "--darkness", args.darkness,
        ]

        if debug_path:
            cmd += ["--debug-output", str(debug_path)]

        print(f"[{i}/{len(videos)}] processing: {video}")
        start = time.time()

        try:
            subprocess.run(cmd, check=True)
            status = "success"
        except subprocess.CalledProcessError:
            status = "failed"

        seconds = round(time.time() - start, 2)
        results.append({"video": str(video), "output": str(output_path), "status": status, "seconds": seconds})
        print(f"[{i}/{len(videos)}] {status} in {seconds}s")

    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "output", "status", "seconds"])
        writer.writeheader()
        writer.writerows(results)

    print("Batch done.")
    print(f"Log: {log_csv}")

if __name__ == "__main__":
    main()
