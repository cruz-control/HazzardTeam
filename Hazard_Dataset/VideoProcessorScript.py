import cv2
import os
import csv

def process_video(video_path, video_output_folder, frame_count=150, output_size=256):
    """
    Processes a single video by extracting the last `frame_count` frames, cropping, resizing, and saving them quickly.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(max(0, total_frames - frame_count), total_frames))

    # Ensure output folder exists
    os.makedirs(video_output_folder, exist_ok=True)

    # Prepare CSV data
    csv_file_path = os.path.join(video_output_folder, 'classification.csv')
    csv_data = []

    for idx, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Frame {frame_idx} could not be read.")
            continue

        # Crop the largest square
        height, width, _ = frame.shape
        square_size = min(height, width)
        crop_x = (width - square_size) // 2
        crop_y = (height - square_size) // 2
        cropped_frame = frame[crop_y:crop_y + square_size, crop_x:crop_x + square_size]

        # Resize to output size
        resized_frame = cv2.resize(cropped_frame, (output_size, output_size), interpolation=cv2.INTER_AREA)

        # Save the resized frame
        frame_filename = f"frame_{idx}.jpg"
        frame_path = os.path.join(video_output_folder, frame_filename)
        cv2.imwrite(frame_path, resized_frame)

        # Add to CSV
        csv_data.append([frame_filename, 0])

    cap.release()

    # Write CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)

    print(f"Finished processing {os.path.basename(video_path)}")

def process_videos_in_folder(input_folder, output_folder, frame_count=150, output_size=256):
    """
    Processes all videos in the input folder sequentially, optimized for speed.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all video files in the folder
    video_files = [
        video_filename for video_filename in os.listdir(input_folder)
        if video_filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    ]

    if not video_files:
        print("No video files found in the input folder.")
        return

    for idx, video_filename in enumerate(video_files, start=1):
        video_path = os.path.join(input_folder, video_filename)
        video_output_folder = os.path.join(output_folder, str(idx))

        print(f"Processing video {idx}/{len(video_files)}: {video_filename}")
        process_video(video_path, video_output_folder, frame_count, output_size)

    print(f"All videos processed. Outputs saved in {output_folder}")

if __name__ == '__main__':
    # User inputs
    input_folder = input("Enter the path for your desired input directory: ").strip()
    output_folder = input("Enter the path for your desired output directory: ").strip()
    process_videos_in_folder(input_folder, output_folder)
