"""Audio-visual helper functions."""
import cv2
import numpy as np
import torch


def save_video_with_audio(video, audio, output_path):
    """
    Saves a video file with audio.
    
    Args:
        video (np.ndarray): Video frames.
        audio (np.ndarray): Audio samples.
        output_path (str): Output path.
    """
    # create video writer
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video.shape[2], video.shape[1]))
    # write the image frames to the video
    for frame in video:
        video_writer.write(frame)
    # add the audio data to the video
    video_writer.write(audio)
    # release the VideoWriter object
    video_writer.release()


def save_video_from_image_sequence_and_audio(sequence, audio, save_path, video_fps=15, audio_fps=22100):
    from moviepy.editor import VideoClip, AudioClip, ImageSequenceClip
    from moviepy.audio.AudioClip import AudioArrayClip
    
    assert isinstance(sequence, list) and isinstance(audio, (np.ndarray, torch.Tensor))
    assert len(audio.shape) == 2 and audio.shape[1] in [1, 2]
    
    video_duration = len(sequence) / video_fps
    audio_duration = len(audio) / audio_fps
    # print(f"Video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s")
    assert video_duration == audio_duration, \
        f"Video duration ({video_duration}) and audio duration ({audio_duration}) do not match."

    video_clip = ImageSequenceClip(sequence, fps=video_fps)
    audio_clip = AudioArrayClip(audio, fps=audio_fps)
    video_clip = video_clip.set_audio(audio_clip)
    # video_clip.write_videofile(save_path, verbose=True, logger=None, fps=video_fps, audio_fps=audio_fps)
    video_clip.write_videofile(save_path, verbose=False, logger=None)
