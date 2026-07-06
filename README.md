# Face Blur Tool

A Python tool for automatically detecting and blurring faces in videos.

This tool supports batch processing. Add videos to an input folder, run one command, and the blurred videos will be saved to an output folder.

## Clone

Clone the repository:

    git clone https://github.com/rishitat-tech/face-blur-tool.git
    cd face-blur-tool

## Install

Install the required packages:

    pip install -r requirements.txt

If needed, install the main dependencies manually:

    pip install opencv-python numpy tqdm insightface onnxruntime

## How to Use

Create an input folder:

    mkdir -p input_videos

Add videos to:

    input_videos/

Example:

    input_videos/
      video1.mp4
      video2.mp4
      video3.mp4

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

The tool processes videos one by one from the input folder.

There is no fixed hard limit for how many videos can be added in one batch. The practical limit depends on video length, resolution, storage, memory, and machine performance.

Recommended batch sizes:

- Small batch: 5-20 videos
- Normal batch: 50-200 videos
- Large batch: 200-500 videos
- Thousands of videos: split into multiple folders

Example:

    input_batch_001/
    input_batch_002/
    input_batch_003/

For large batches, use --skip-existing so completed videos are not processed again.

## Recommended Settings

Use these default settings:

    --det-thresh 0.50
    --det-size 1024
    --box-scale 1.25
    --memory-frames 3
    --darkness 0.65

## Notes

- Keep input videos in a dedicated folder.
- Use --skip-existing for large batches.
- Debug videos are useful for reviewing detection and blur coverage.
- Processing time depends on video length, resolution, and machine performance.
