"""Misc utils."""
import os
from shared.utils.log import tqdm_iterator
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def ignore_warnings(type="ignore"):
    import warnings
    warnings.filterwarnings(type)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_youtube_video(youtube_id, ext='mp4', resolution="360p", **kwargs):
    import pytube
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"
    yt = pytube.YouTube(video_url)
    try:
        streams = yt.streams.filter(
            file_extension=ext, res=resolution, progressive=True, **kwargs,
        )
        # streams[0].download(output_path=save_dir, filename=f"{video_id}.{ext}")
        streams[0].download(output_path='/tmp', filename='sample.mp4')
    except:
        print("Failed to download video: ", video_url)
        return None
    return "/tmp/sample.mp4"


def check_audio(video_path):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    try:
        return VideoFileClip(video_path).audio is not None
    except:
        return False


def check_audio_multiple(video_paths, n_jobs=8):
    """Parallelly check if videos have audio"""
    iterator = tqdm_iterator(video_paths, desc="Checking audio")
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
            delayed(check_audio)(video_path) for video_path in iterator
        )


def num_trainable_params(model):
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    model_name = model.__class__.__name__
    print(f"::: Number of trainable parameters in {model_name}: {np.round(n_params / 1e6, 3)}M")


def num_params(model):
    n_params = sum([p.numel() for p in model.parameters()])
    model_name = model.__class__.__name__
    print(f"::: Number of total parameters in {model_name}: {np.round(n_params / 1e6, 3)}M")