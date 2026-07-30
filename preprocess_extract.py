"""
Stages I–II — Frame preprocessing and spatial/temporal feature extraction
=========================================================================
Paper: "Enhanced inter-frame video forgery detection using convolutional
network and stacking ensemble"
       Multimedia Tools and Applications (2026) 85:497
       https://doi.org/10.1007/s11042-026-21684-x

Implements:
  Stage I  — grayscale conversion, NLM denoising, sharpening, normalisation,
             and optional geometric augmentation (flip / rotate / crop)
  Stage II — edge-difference D, Farneback optical flow O, and HOG H features
             (Eqs. 1–4 in the paper), interpolated back to the original frame
             count so every video yields equal-length feature series

Pipeline order
--------------
1. annotate.py
2. preprocess_extract.py   ← you are here
3. tcn_feat.py
4. ensemble_class.py

Inputs
------
Annotation JSON(s) under ``output/annotations/`` produced by ``annotate.py``.

Outputs
-------
- ``output/all_preprocess_feat.json`` when ``split_data=True``
- ``output/{train,test,val}_preprocess_feat.json`` otherwise

Each entry maps a video path to::

    {"edge": [...], "optical_flow": [...], "hog": [...]}
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from imgaug import augmenters as iaa
from scipy.interpolate import interp1d
from skimage.feature import hog


class VideoFrameExtractor:
    """Preprocess video frames and extract D / O / H feature sequences."""

    def __init__(self, parent_folder: str, config: dict):
        """
        Parameters
        ----------
        parent_folder : str
            Kept for API symmetry with earlier pipeline stages (videos are
            loaded from annotation JSON paths, not re-scanned here).
        config : dict
            - augment_data : enable flip / rotate / crop augmentation
            - split_data   : read all_annotate.json vs train/test/val files
        """
        self.parent_folder = parent_folder
        self.video_files = []
        self.annotations = {}
        self.augment_data = config.get("augment_data", True)
        self.split_data = config.get("split_data", False)
        print(f"split_data is set to: {self.split_data}")

    def get_all_videos(self) -> None:
        """Load video paths and labels from annotation JSON files."""
        if self.split_data:
            annotation_files = ["all_annotate.json"]
        else:
            annotation_files = [
                "train_annotate.json",
                "test_annotate.json",
                "val_annotate.json",
            ]

        for annotation_file in annotation_files:
            annotation_path = os.path.join("output", "annotations", annotation_file)
            print(f"Looking for annotations in {annotation_path}")

            if os.path.exists(annotation_path):
                with open(annotation_path, "r") as f:
                    annotations = json.load(f)
                    for video_name in annotations:
                        if "video_path" in annotations[video_name]:
                            self.video_files.append(annotations[video_name]["video_path"])
                            self.annotations[video_name] = annotations[video_name]
            else:
                print(f"Annotations file not found: {annotation_path}")

    def extract_frames(self) -> dict:
        """
        Process every annotated video in a thread pool.

        Returns
        -------
        dict
            Mapping video_path → feature dict with list-serialised arrays.
        """
        all_features = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_video = {
                executor.submit(self.process_video, video_path): video_path
                for video_path in self.video_files
            }
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_name, frames, features = future.result()
                    if frames:
                        # JSON-serialise numpy arrays for the next pipeline stage
                        all_features[video_path] = {k: v.tolist() for k, v in features.items()}
                    else:
                        print(f"No frames extracted for video: {video_path}")
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")

        return all_features

    def process_video(self, video_path: str):
        """Load one video, preprocess its frames, and compute D / O / H."""
        try:
            info = self.video_info(video_path)
            frames = info["frames"]
            features = self.extract_features(frames, info["exact_frame_count"])
            return video_path, frames, features
        except Exception as e:
            print(f"Error in process_video for {video_path}: {e}")
            raise e

    def video_info(self, video_file: str) -> dict:
        """Open a video, read metadata, and return preprocessed sampled frames."""
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Couldn't open video file: {video_file}")

        exact_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = self.extract_video_frames(cap, exact_frame_count)
        cap.release()
        return {
            "exact_frame_count": exact_frame_count,
            "frame_rate": frame_rate,
            "resolution": (width, height),
            "frames": frames,
        }

    def extract_video_frames(self, cap, frame_count: int, every_nth_frame: int = 5) -> list:
        """
        Sample every N-th frame, preprocess, and optionally augment.

        Subsampling reduces cost while still covering the temporal span; later
        interpolation restores features to the original frame count.
        """
        frames = []
        for i in range(0, frame_count, every_nth_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = self.preprocess_frame(frame)
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame at position {i}")
        if self.augment_data:
            frames = self.augment_frames(frames)
        return frames

    def preprocess_frame(self, frame):
        """
        Stage I preprocessing (paper §2.2.1).

        RGB → grayscale → non-local means denoise → sharpen → normalise.
        Removes post-manipulation artefacts (noise, inpainting residue, etc.)
        that forgers often apply to hide temporal cuts.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fast NLM denoising [Vignesh et al., 2009] — paper Eq. Stage I
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        # 3×3 Laplacian-style sharpening kernel to reinforce edges
        sharpened = cv2.filter2D(
            denoised, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        )
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    def augment_frames(self, frames: list) -> list:
        """Geometric augmentation: horizontal flip, small rotation, mild crop."""
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-10, 10)),
                iaa.Crop(percent=(0, 0.1)),
            ]
        )
        return seq(images=frames)

    def extract_features(self, frames: list, original_frame_count: int) -> dict:
        """
        Stage II feature extraction (paper §2.2.2, Eqs. 1–4).

        Returns three complementary cues that together cover edge, motion and
        structure so a miss in one modality can still be caught by another.
        """
        features = {}
        # D — Canny edge difference between consecutive frames (Eq. 1)
        features["edge"] = self.interpolate_features(
            self.calculate_edge_differences_function(frames), original_frame_count
        )
        # O — Farneback optical-flow magnitude sum (Eq. 2)
        features["optical_flow"] = self.interpolate_features(
            self.calculate_optical_flow_function(frames), original_frame_count
        )
        # H — summed HOG descriptor energy per frame (Eq. 3)
        features["hog"] = self.interpolate_features(
            self.calculate_hog_features(frames), original_frame_count
        )
        return features

    def calculate_edge_differences_function(self, frames: list) -> list:
        """
        Frame-edge difference D (Eq. 1).

        Abrupt peaks indicate insertion / deletion boundaries where Canny edges
        change discontinuously between neighbouring frames.
        """
        edge_diffs = []
        for i in range(len(frames) - 1):
            edges_curr = cv2.Canny(frames[i], 100, 200)
            edges_next = cv2.Canny(frames[i + 1], 100, 200)
            diff = np.sum(np.abs(edges_curr - edges_next))
            edge_diffs.append(diff)
        return edge_diffs

    def calculate_optical_flow_function(self, frames: list) -> list:
        """
        Optical-flow magnitude O (Eq. 2) via Farneback dense flow [2003].

        Captures broken motion / sudden stops that survive compression better
        than pure edge cues.
        """
        flow_diffs = []
        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            diff = np.sum(np.abs(flow))
            flow_diffs.append(diff)
        return flow_diffs

    def calculate_hog_features(self, frames: list) -> list:
        """
        Histogram of Oriented Gradients H (Eq. 3) [Dalal & Triggs, 2005].

        Especially useful for duplication, where motion cues are weaker but
        repeated spatial texture/gradient patterns remain distinctive.
        """
        hog_features = []
        for frame in frames:
            features, _ = hog(
                frame,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualize=True,
            )
            hog_features.append(np.sum(features))
        return hog_features

    def interpolate_features(self, features: list, original_frame_count: int) -> list:
        """
        Resample a subsampled feature series onto the full frame timeline.

        Linear interpolation aligns D / O / H with the original video length so
        the TCN stage receives consistent temporal resolution across clips.
        """
        x = np.linspace(0, original_frame_count - 1, len(features))
        f = interp1d(x, features, kind="linear")
        x_new = np.arange(original_frame_count)
        return f(x_new)


def main():
    """
    Entry point — reads annotations, extracts features, writes JSON outputs.

    Keep ``split_data`` consistent with the value used in ``annotate.py``.
    """
    config = {
        "augment_data": True,
        "split_data": True,
    }
    parent_folder = "/path/to/dataset/merged"
    extractor = VideoFrameExtractor(parent_folder, config)
    extractor.get_all_videos()
    all_features = extractor.extract_frames()

    if not os.path.exists("output"):
        os.makedirs("output")

    if config["split_data"]:
        with open(os.path.join("output", "all_preprocess_feat.json"), "w") as f:
            json.dump(all_features, f)
    else:
        # Bucket features by path substring when using explicit splits
        with open(os.path.join("output", "train_preprocess_feat.json"), "w") as f:
            json.dump({k: v for k, v in all_features.items() if "train" in k}, f)
        with open(os.path.join("output", "test_preprocess_feat.json"), "w") as f:
            json.dump({k: v for k, v in all_features.items() if "test" in k}, f)
        with open(os.path.join("output", "val_preprocess_feat.json"), "w") as f:
            json.dump({k: v for k, v in all_features.items() if "val" in k}, f)


if __name__ == "__main__":
    main()
