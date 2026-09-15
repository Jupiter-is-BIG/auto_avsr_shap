import os

import cv2
import soundfile as sf
import torch
import torchvision


def load_wav(path):
    """
    torchaudio.load()-compatible WAV reader: returns (waveform, sample_rate)
    with waveform shape (channels, num_samples), float32, normalized to
    [-1, 1] -- matching torchaudio.load(path, normalize=True)'s contract.
    Implemented via soundfile instead of torchaudio's torchcodec-based
    backend, since this environment's torch (2.14.0) / torchaudio (2.11.0)
    aren't from the same release train and torchcodec isn't installed;
    soundfile has no such dependency.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    waveform = torch.from_numpy(data.T).contiguous()  # (C, T)
    return waveform, sample_rate


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def _load_video_cv2(path):
    """
    Fallback video reader for torchvision releases that dropped
    torchvision.io.read_video (video decoding moved to the separate
    torchcodec library). Matches load_video's contract: T x C x H x W, RGB.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(torch.from_numpy(frame_bgr[..., ::-1].copy()))
    finally:
        cap.release()
    vid = torch.stack(frames)  # T x H x W x C, RGB, uint8
    return vid.permute((0, 3, 1, 2))


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    if hasattr(torchvision.io, "read_video"):
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        vid = vid.permute((0, 3, 1, 2))
        return vid
    return _load_video_cv2(path)


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = load_wav(path[:-4] + ".wav")
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            rel_path = rel_path.replace("lrs3_video_seg16s", "lrs3_video_seg24s")
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    torch.tensor([int(_) for _ in token_id.split()]),
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
