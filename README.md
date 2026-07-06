# Face Blur Tool

A Python-based tool for automatically detecting and blurring faces in videos. It supports batch processing, debug video generation, and CSV logging.

## Usage

This tool detects faces in videos and applies blur automatically. It is designed for batch processing, so multiple videos can be placed in one input folder and processed together.

## Recommended Configuration

```bash
--det-thresh 0.50
--det-size 1024
--box-scale 1.25
--memory-frames 3
--darkness 0.65
```

## Run Batch Processing

Place videos inside the `input_videos` folder:

```text
input_videos/
  video1.mp4
  video2.mp4
  video3.mp4
```

Then run:

```bash
python batch_blur_memory.py \
  --input-dir input_videos \
  --output-dir batch_outputs/blurred_final_clean \
  --debug-dir batch_outputs/debug_final_clean \
  --log-csv batch_outputs/log_final_clean.csv \
  --det-thresh 0.50 \
  --det-size 1024 \
  --box-scale 1.25 \
  --memory-frames 3 \
  --darkness 0.65 \
  --skip-existing
```

Processed videos are saved in:

```text
batch_outputs/blurred_final_clean/
```

Debug videos are saved in:

```text
batch_outputs/debug_final_clean/
```

The CSV processing log is saved at:

```text
batch_outputs/log_final_clean.csv
```

## Batch Processing

There is no fixed hard limit on how many videos can be added to a batch. The tool processes videos one by one from the input folder.

The practical batch size depends on:

- video length
- video resolution
- available storage
- CPU/GPU speed
- available memory
- available processing time

Recommended batch sizes:

| Use Case | Recommended Batch Size |
|---|---:|
| Small batch | 5-20 videos |
| Normal batch | 50-200 videos |
| Large batch | 200-500 videos |
| Very large dataset | Split into multiple batches |

For thousands of videos, split them into smaller folders:

```text
input_batch_001/
input_batch_002/
input_batch_003/
```

The `--skip-existing` flag is useful for large batches because it avoids reprocessing videos that already have outputs.

## Backend

The backend is handled by Python scripts.

Main backend files:

```text
batch_blur_memory.py
blur_all_faces_memory.py
```

### Backend Flow

```text
Input videos
   ↓
batch_blur_memory.py
   ↓
blur_all_faces_memory.py
   ↓
Blurred output videos
   ↓
Debug videos and CSV log
```

### Backend Responsibilities

The backend handles:

- reading videos from an input folder
- detecting faces frame by frame
- applying blur to detected faces
- keeping short temporal memory for smoother blur
- writing processed videos
- writing debug videos
- generating a CSV processing log
- skipping already processed files when `--skip-existing` is used

## Frontend

The current project is backend/script-based. A frontend can be added on top of the backend to make the tool easier to use.

A frontend can provide:

- video upload
- batch folder selection
- processing status
- progress tracking
- output preview
- debug preview
- download links
- configurable blur settings

### Recommended Frontend Defaults

| Frontend Setting | Backend Argument | Default |
|---|---|---:|
| Detection threshold | `--det-thresh` | `0.50` |
| Detection size | `--det-size` | `1024` |
| Blur box scale | `--box-scale` | `1.25` |
| Memory frames | `--memory-frames` | `3` |
| Darkness | `--darkness` | `0.65` |

The frontend should call the backend script with these default values unless the user changes them.

## Output Structure

```text
batch_outputs/
  blurred_final_clean/
    video1_blurred.mp4
    video2_blurred.mp4
  debug_final_clean/
    video1_debug.mp4
    video2_debug.mp4
  log_final_clean.csv
```

## Notes

- Use `--skip-existing` for large batches to avoid reprocessing completed videos.
- Keep input videos in a dedicated folder before running the batch command.
- Debug videos are useful for reviewing detections and blur coverage.
- Processing time depends on video count, video length, resolution, and machine performance.
