
import cv2


def get_total_frames(vidcap):
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in video {num_frames}")
    return num_frames


def get_video_height_width(vidcap):
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(f"Video size: {width} x {height}")
    return height, width