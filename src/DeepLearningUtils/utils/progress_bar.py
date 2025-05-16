
from tqdm import tqdm

def create_progress_bar(num_frames, label="Train"):
    """
    Creates a progress bar that increments with each frame processed

    Parameters
    ----------
    num_frames: int

    Returns
    -------

    """
    iterable = enumerate(range(0, num_frames))
    progress = tqdm(
        iterable, desc=label, total=num_frames, ascii=True, leave=True, position=0
    )
    iterable = progress
    return iterable
