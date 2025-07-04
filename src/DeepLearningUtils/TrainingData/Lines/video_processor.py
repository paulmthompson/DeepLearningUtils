import cv2
import numpy as np
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from tqdm import tqdm
import keras

from src.DeepLearningUtils.DataStructures.Masks.sparse_mask import SparseVideoMask
from src.DeepLearningUtils.utils.video.video_opencv import get_total_frames, get_video_height_width


def process_video_batch(
    video_paths: List[str],
    model,
    batch_size: int,
    threshold: float = 0.5,
    output_format: str = "hdf5",
    log_file: Optional[str] = None
) -> None:
    """
    Process a batch of videos through a line segmentation model.

    Parameters
    ----------
    video_paths : List[str]
        List of paths to video files to process
    model : tf.keras.Model
        Trained model for line segmentation
    batch_size : int
        Number of frames to process in each batch
    threshold : float
        Probability threshold for line detection
    output_format : str
        Format for output files (currently only "hdf5" supported)
    log_file : Optional[str]
        Path to log file. If None, will create a log file in the same directory
        as the first video with timestamp

    Notes
    -----
    - Creates a log file with processing details and any errors
    - Shows progress for both individual videos and overall progress
    - Estimates remaining time based on average processing speed
    - Places output files in the same directory as input videos
    - Skips videos that can't be opened and logs the error
    - Uses SparseVideoMask for efficient storage of line masks
    """
    # Setup logging
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(Path(video_paths[0]).parent / f"video_processing_{timestamp}.log")
    
    log_handle = open(log_file, "w")
    log_handle.write(f"Video Processing Log - Started at {datetime.now()}\n")
    log_handle.write("=" * 80 + "\n\n")

    # Get model input dimensions
    input_shape = model.input_shape
    model_input_height = input_shape[1]
    model_input_width = input_shape[2]
    model_input_channels = input_shape[3]

    # Process each video
    total_videos = len(video_paths)
    total_frames_processed = 0
    total_processing_time = 0

    for video_idx, video_path in enumerate(video_paths, 1):
        video_path = str(video_path)  # Ensure string path
        log_handle.write(f"\nProcessing video {video_idx}/{total_videos}: {video_path}\n")
        log_handle.write("-" * 80 + "\n")

        try:
            # Open video
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            height, width = get_video_height_width(vidcap)
            height, width = int(height), int(width)  # Convert to integers for array dimensions
            num_frames = get_total_frames(vidcap)
            
            # Calculate scaling factors
            scale_height = height / model_input_height
            scale_width = width / model_input_width

            log_handle.write(f"Video dimensions: {width}x{height}\n")
            log_handle.write(f"Number of frames: {num_frames}\n")
            log_handle.write(f"Scaling factors: height={scale_height:.2f}, width={scale_width:.2f}\n")

            # Initialize progress bar
            progress = tqdm(
                total=num_frames,
                desc=f"Video {video_idx}/{total_videos}",
                position=0,
                leave=True
            )

            # Initialize batch
            prediction_batch = np.zeros(
                (batch_size, height, width, model_input_channels),
                dtype=np.uint8
            )
            batch_counter = 0
            frame_counter = 0

            # Initialize mask storage
            video_masks = SparseVideoMask(
                threshold=threshold,
                video_height=model_input_height,
                video_width=model_input_width
            )

            # Process frames
            start_time = time.time()
            while True:
                success, image = vidcap.read()
                if not success:
                    break

                prediction_batch[batch_counter] = image
                batch_counter += 1

                if batch_counter == batch_size:
                    # Process batch
                    labels = model.predict_on_batch(prediction_batch)
                    frame_offset = frame_counter
                    video_masks.add_frames(labels[:, :, :, 0], frame_offset)
                    
                    # Update counters
                    frame_counter += batch_size
                    batch_counter = 0
                    progress.update(batch_size)

            # Process remaining frames
            if batch_counter > 0:
                labels = model.predict_on_batch(prediction_batch[:batch_counter])
                video_masks.add_frames(labels[:, :, :, 0], frame_counter)
                progress.update(batch_counter)

            # Calculate processing time
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            total_frames_processed += num_frames

            # Save results
            output_path = Path(video_path).parent / "line_masks.h5"
            video_masks.to_hdf5(str(output_path))

            # Log success
            log_handle.write(f"Successfully processed {num_frames} frames\n")
            log_handle.write(f"Processing time: {processing_time:.2f} seconds\n")
            log_handle.write(f"Average time per frame: {processing_time/num_frames:.3f} seconds\n")
            log_handle.write(f"Saved masks to {output_path}\n")

            # Estimate remaining time
            if video_idx < total_videos:
                avg_time_per_frame = total_processing_time / total_frames_processed
                remaining_frames = sum(get_total_frames(cv2.VideoCapture(str(p))) for p in video_paths[video_idx:])
                est_remaining_time = avg_time_per_frame * remaining_frames
                log_handle.write(f"Estimated time remaining: {timedelta(seconds=int(est_remaining_time))}\n")

        except Exception as e:
            log_handle.write(f"ERROR processing video: {str(e)}\n")
            log_handle.write(f"Stack trace:\n{traceback.format_exc()}\n")

        finally:
            if 'vidcap' in locals():
                vidcap.release()
            progress.close()

    # Write summary
    try:
        log_handle.write("\n" + "=" * 80 + "\n")
        log_handle.write("Processing Summary\n")
        log_handle.write(f"Total videos processed: {total_videos}\n")
        log_handle.write(f"Total frames processed: {total_frames_processed}\n")
        
        if total_frames_processed > 0:
            log_handle.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
            log_handle.write(f"Average time per frame: {total_processing_time/total_frames_processed:.3f} seconds\n")
        else:
            log_handle.write("No frames were successfully processed\n")
            
        log_handle.write(f"Log file: {log_file}\n")
        log_handle.write(f"Completed at: {datetime.now()}\n")
    except Exception as e:
        print(f"Error writing summary to log: {str(e)}")
    finally:
        log_handle.close()
        print(f"\nProcessing complete. Log file: {log_file}")


def prediction_loop(
    batch_size: int,
    model,
    video_path: str,
    threshold: float = 0.5,
    num_frames: Optional[int] = None,
) -> SparseVideoMask:
    """
    Process a single video through a line segmentation model.

    Parameters
    ----------
    batch_size : int
        Number of frames to process in each batch
    model : keras.Model
        Trained model for line segmentation
    video_path : str
        Path to video file
    threshold : float
        Probability threshold for line detection
    num_frames : Optional[int]
        Number of frames to process. If None, processes entire video.

    Returns
    -------
    SparseVideoMask
        Object containing detected line masks
    """
    input_shape = model.input_shape
    model_input_height = input_shape[1]
    model_input_width = input_shape[2]
    model_input_channels = input_shape[3]

    vidcap = cv2.VideoCapture(video_path)
    height, width = get_video_height_width(vidcap)

    if num_frames is None:
        num_frames = get_total_frames(vidcap)

    prediction_batch = np.zeros(
        (batch_size, height, width, model_input_channels),
        dtype=np.uint8
    )
    batch_counter = 0
    frame_counter = 0

    video_masks = SparseVideoMask(
        threshold=threshold,
        video_height=model_input_height,
        video_width=model_input_width
    )

    for i in tqdm(range(num_frames), desc="Processing frames"):
        success, image = vidcap.read()
        if not success:
            break

        prediction_batch[batch_counter] = image
        batch_counter += 1

        if batch_counter == batch_size:
            labels = model.predict_on_batch(prediction_batch)
            frame_offset = i - batch_size + 1
            video_masks.add_frames(labels[:, :, :, 0], frame_offset)
            batch_counter = 0
            frame_counter += batch_size

    if batch_counter > 0:
        labels = model.predict_on_batch(prediction_batch[:batch_counter])
        video_masks.add_frames(labels[:, :, :, 0], frame_counter)

    vidcap.release()
    return video_masks 