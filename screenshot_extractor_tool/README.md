# Video Screenshot Extractor

A Python script that extracts screenshots from video files at regular intervals and saves them to a folder.

## Features

- Extract screenshots at customizable intervals (default: 5 seconds)
- Automatic folder creation for screenshots
- Detailed video information display (duration, FPS, total frames)
- Timestamps in filenames for easy identification
- Command-line interface with flexible options
- Error handling for invalid video files

## Requirements

- Python 3.6+
- OpenCV (cv2)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install OpenCV directly:
```bash
pip install opencv-python
```

## Usage

### Command Line Interface

```bash
# Basic usage - extract screenshots every 5 seconds
python video_screenshot_extractor.py training_video.mp4

# Custom output folder
python video_screenshot_extractor.py training_video.mp4 --output_folder my_screenshots

# Custom interval (every 10 seconds)
python video_screenshot_extractor.py training_video.mp4 --interval 10

# Custom folder and interval
python video_screenshot_extractor.py training_video.mp4 -o frames -i 2
```

### Programmatic Usage

```python
from video_screenshot_extractor import extract_screenshots

# Extract screenshots every 5 seconds
count = extract_screenshots("training_video.mp4", "screenshots", 5)
print(f"Extracted {count} screenshots")
```

### Example Script

Run the included example:
```bash
python example_usage.py
```

## Command Line Arguments

- `video_path`: Path to the input video file (required)
- `--output_folder, -o`: Output folder for screenshots (default: "screenshots")
- `--interval, -i`: Interval between screenshots in seconds (default: 5)

## Output

Screenshots are saved as JPEG files with the following naming convention:
```
{video_name}_screenshot_{number}_{timestamp}s.jpg
```

Example:
```
training_video_screenshot_0001_0.0s.jpg
training_video_screenshot_0002_5.0s.jpg
training_video_screenshot_0003_10.0s.jpg
```

## Supported Video Formats

The script supports all video formats that OpenCV can read, including:
- MP4
- AVI
- MOV
- MKV
- FLV
- WMV
- And many others

## Example Output

```
Video info:
  - Duration: 120.45 seconds
  - FPS: 30.00
  - Total frames: 3614
  - Screenshot interval: 5 seconds

Screenshot 1: training_video_screenshot_0001_0.0s.jpg (at 0.0s)
Screenshot 2: training_video_screenshot_0002_5.0s.jpg (at 5.0s)
Screenshot 3: training_video_screenshot_0003_10.0s.jpg (at 10.0s)
...

Extraction complete!
Total screenshots extracted: 25
Screenshots saved to: screenshots
```

## Error Handling

The script includes comprehensive error handling for:
- Missing video files
- Corrupted or unreadable video files
- Invalid intervals
- File system permissions
- Failed screenshot saves

## Notes

- Screenshots are saved as high-quality JPEG files
- The script maintains the original video resolution
- Progress is displayed in real-time during extraction
- The output folder is created automatically if it doesn't exist