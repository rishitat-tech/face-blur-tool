# Face Blur Tool

A Python backend tool for automatically detecting and blurring faces in videos.

## How to Use

Add videos to:

    input_videos/

Then run:

    python batch_blur_memory.py --input-dir input_videos --output-dir batch_outputs/blurred_final_clean --debug-dir batch_outputs/debug_final_clean --log-csv batch_outputs/log_final_clean.csv --det-thresh 0.50 --det-size 1024 --box-scale 1.25 --memory-frames 3 --darkness 0.65 --skip-existing

## Output

Blurred videos:

    batch_outputs/blurred_final_clean/

Debug videos:

    batch_outputs/debug_final_clean/

CSV log:

    batch_outputs/log_final_clean.csv

## Batch Processing

This tool is used for batch processing. You can add multiple videos to the input folder and the tool will process them one by one.

There is no fixed hard limit for how many videos can be added in one batch. The practical limit depends on video length, resolution, storage, and machine performance.

Recommended:
- Small batch: 5-20 videos
- Normal batch: 50-200 videos
- Large batch: 200-500 videos
- Thousands of videos: split into multiple folders

Example:

    input_batch_001/
    input_batch_002/
    input_batch_003/

Use --skip-existing so completed videos are not processed again.

## Backend

The backend is the Python processing layer.

Main backend files:

    batch_blur_memory.py
    blur_all_faces_memory.py

The backend handles:
- reading videos from the input folder
- detecting faces frame by frame
- blurring detected faces
- saving blurred videos
- saving debug videos
- writing the CSV log
- skipping completed videos when --skip-existing is used

## Frontend

A frontend is not required to run this tool right now.

A frontend can be added later for:
- uploading videos
- selecting batches
- starting processing
- viewing progress
- previewing outputs
- downloading blurred videos

If a frontend is added, it should call the backend using the command shown above.
