
import os
import subprocess


def extract_frames_from_videos(training_path, output_path, frame_size=(256, 256)):
    # Walk through the directory structure and collect video file names
    video_files = []
    for root, _, files in os.walk(training_path):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} video files")

    #I want to create a progress bar for this
    vid_count = 0
    total_vids = len(video_files)

    # Create corresponding directory structure and extract frames
    for video_file in video_files:
        # Create a new folder for each video with its name as the folder name
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        relative_path = os.path.relpath(video_file, training_path)
        new_folder_path = os.path.join(output_path, os.path.dirname(relative_path), video_name)

        # Check if the folder already exists and contains images
        if os.path.exists(new_folder_path) and len(os.listdir(new_folder_path)) > 0:
            #print(f"Skipping {video_name}, frames already extracted")
            total_vids -= 1
            continue

        os.makedirs(new_folder_path, exist_ok=True)

        extract_frames_with_ffmpeg(video_file, new_folder_path, frame_size)
        print(f"Extracted frames from {video_name} {vid_count}/{total_vids}")
        vid_count += 1


def extract_frames_with_ffmpeg(video_file, output_folder, frame_size=(256, 256)):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the ffmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-v', 'quiet',
        '-stats',
        '-i', video_file,
        '-vf', f'scale={frame_size[0]}:{frame_size[1]},format=gray',
        '-q:v', '2',  # Quality level for JPEG (2 is high quality, 31 is low quality)
        os.path.join(output_folder, 'frame_%04d.jpg')
    ]

    # Run the ffmpeg command
    subprocess.run(ffmpeg_command, check=True)