#!/usr/bin/env python3
"""
Video Screenshot Extractor

This script takes screenshots from a video file at regular intervals (default: 5 seconds)
and saves them to a specified output folder.

Requirements:
- opencv-python (cv2)
- os, argparse (built-in modules)

Usage:
python video_screenshot_extractor.py input_video.mp4 --output_folder screenshots --interval 5
"""

import cv2
import os
import argparse
import sys
from pathlib import Path


def extract_screenshots(video_path, output_folder, interval_seconds=5):
    """
    Extract screenshots from a video file at regular intervals.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder for screenshots
        interval_seconds (int): Interval between screenshots in seconds

    Returns:
        int: Number of screenshots extracted
    """

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return 0

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video info:")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Screenshot interval: {interval_seconds} seconds")

    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)

    screenshot_count = 0
    frame_number = 0

    # Get video filename without extension for naming screenshots
    video_name = Path(video_path).stem

    while True:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = cap.read()

        if not ret:
            break

        # Generate timestamp for filename
        timestamp_seconds = frame_number / fps

        # Create filename with timestamp
        filename = f"{video_name}_screenshot_{screenshot_count+1:04d}_{timestamp_seconds:.1f}s.jpg"
        filepath = os.path.join(output_folder, filename)

        # Save screenshot
        success = cv2.imwrite(filepath, frame)

        if success:
            screenshot_count += 1
            print(
                f"Screenshot {screenshot_count}: {filename} (at {timestamp_seconds:.1f}s)"
            )
        else:
            print(f"Error: Failed to save screenshot {filename}")

        # Move to next frame position
        frame_number += frame_interval

        # Check if we've reached the end of the video
        if frame_number >= total_frames:
            break

    # Release video capture
    cap.release()

    print(f"\nExtraction complete!")
    print(f"Total screenshots extracted: {screenshot_count}")
    print(f"Screenshots saved to: {output_folder}")

    return screenshot_count


def main():
    """Main function to handle command line arguments and run the extraction."""

    parser = argparse.ArgumentParser(
        description="Extract screenshots from a video file at regular intervals"
    )

    parser.add_argument("video_path", help="Path to the input video file")

    parser.add_argument(
        "--output_folder",
        "-o",
        default="screenshots",
        help="Output folder for screenshots (default: screenshots)",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=5,
        help="Interval between screenshots in seconds (default: 5)",
    )

    args = parser.parse_args()

    # Validate interval
    if args.interval <= 0:
        print("Error: Interval must be greater than 0")
        sys.exit(1)

    # Run extraction
    screenshot_count = extract_screenshots(
        args.video_path, args.output_folder, args.interval
    )

    if screenshot_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
