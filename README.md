# Batch Face Blur Tool

This tool batch-processes videos and blurs detected faces using a dark oval blur.

## Features

- Batch processes multiple videos
- Detects faces with InsightFace
- Applies dark oval blur
- Uses memory to reduce flickering
- Saves optional debug videos

## Install

python -m venv faceblur
source faceblur/bin/activate
pip install -r requirements.txt

## Usage

Put videos into:

input_videos/

Run:

python batch_blur_memory.py --input-dir input_videos --output-dir batch_outputs/blurred --debug-dir batch_outputs/debug --log-csv batch_outputs/log.csv --det-thresh 0.40 --det-size 800 --box-scale 1.20 --memory-frames 4 --darkness 0.65 --skip-existing

## Outputs

Blurred videos: batch_outputs/blurred/
Debug videos: batch_outputs/debug/
Log file: batch_outputs/log.csv

## Open results

On macOS, open the blurred output folder with:

open batch_outputs/blurred

To inspect debug videos with detection boxes:

open batch_outputs/debug

## Tuning

If real faces are missed, lower --det-thresh to 0.35.
If random objects are blurred, increase --det-thresh to 0.45.
If blur flickers, increase --memory-frames to 6.
If blur stays too long, decrease --memory-frames to 3.
