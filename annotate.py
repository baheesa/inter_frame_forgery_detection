"""
Stage 0 — Dataset annotation builder
====================================
Paper: "Enhanced inter-frame video forgery detection using convolutional
network and stacking ensemble"
       Multimedia Tools and Applications (2026) 85:497
       https://doi.org/10.1007/s11042-026-21684-x

Walks a dataset directory tree and writes JSON annotation files that map each
video path to its forgery class. Folder names under each video are treated as
class labels (original / insert / delete / duplicate).

Pipeline order
--------------
1. annotate.py              ← you are here
2. preprocess_extract.py
3. tcn_feat.py
4. ensemble_class.py

Expected folder layout (example)::

    <parent>/
        train|test|val|merged/
            original/*.mp4
            insert/*.mp4
            delete/*.mp4
            duplicate/*.mp4

Outputs (under ``config['output_folder']``)
-------------------------------------------
- ``all_annotate.json`` when ``split_data=True``
- ``train_annotate.json``, ``test_annotate.json``, ``val_annotate.json``
  when ``split_data=False`` and the corresponding folder paths are set
"""

import json
import os


class VideoFrameExtractor:
    """Scan a video corpus and emit forgery-type annotation JSON files."""

    def __init__(self, parent_folder: str, config: dict):
        """
        Parameters
        ----------
        parent_folder : str
            Root directory that will be walked recursively for video files.
        config : dict
            Runtime options:
              - output_folder : where annotation JSONs are written
              - train_folder / test_folder / val_folder : path substrings used
                to bucket videos when split_data is False
              - forgery_types : allowed class-folder names
              - split_data : if True, write a single all_annotate.json;
                otherwise write per-split annotation files
        """
        self.parent_folder = parent_folder
        self.config = config
        self.annotations = {}
        # Paper classes: original + three inter-frame forgery types
        self.default_forgery_types = ["duplicate", "delete", "insert", "original"]

    def create_annotations(self):
        """
        Discover videos under ``parent_folder`` and write annotation JSONs.

        Forgery type is inferred from the immediate parent folder name of each
        video (e.g. ``.../insert/clip.mp4`` → forgery_type = ``insert``).
        """
        # Ensure results directory exists
        output_dir = self.config.get("output_folder", "output")
        os.makedirs(output_dir, exist_ok=True)

        annotation_files = {
            "train": os.path.join(output_dir, "train_annotate.json"),
            "test": os.path.join(output_dir, "test_annotate.json"),
            "val": os.path.join(output_dir, "val_annotate.json"),
            "all": os.path.join(output_dir, "all_annotate.json"),
        }

        # Optional path substrings that identify train / test / val splits
        folders = {
            "train": self.config.get("train_folder"),
            "test": self.config.get("test_folder"),
            "val": self.config.get("val_folder"),
        }

        forgery_types = self.config.get("forgery_types", self.default_forgery_types)

        # Fresh run: remove stale annotation files so results stay consistent
        for key, path in annotation_files.items():
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted existing {path}")

        all_annotations = {}
        folder_annotations = {key: {} for key in folders if folders[key]}

        # Collect all supported video containers under the parent tree
        video_files = []
        for subdir, dirs, files in os.walk(self.parent_folder):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".AVI", ".MOV")):
                    video_files.append(os.path.join(subdir, file))

        print(f"Found {len(video_files)} videos in the directory structure.")

        for video_file in video_files:
            video_name = os.path.basename(video_file)
            # Class label = name of the folder that directly contains the video
            last_folder = os.path.basename(os.path.dirname(video_file))

            try:
                new_annotation = {
                    "forgery_type": last_folder if last_folder in forgery_types else "unknown",
                    "video_name": video_name,
                    "video_path": video_file,
                }

                all_annotations[video_file] = new_annotation

                # Also bucket into train/test/val if path contains the configured folder
                for key, folder in folders.items():
                    if folder and folder in video_file:
                        folder_annotations[key][video_file] = new_annotation

            except Exception as e:
                print(f"Error processing video {video_file} in create_annotations: {e}")

        split_data = self.config.get("split_data", False)

        if split_data:
            # Single merged annotation file (typical for balanced public datasets)
            if all_annotations:
                with open(annotation_files["all"], "w") as f:
                    json.dump(all_annotations, f, indent=4)
                print(f"All annotations saved to {annotation_files['all']}")
        else:
            # Separate annotation files for predefined train / test / val folders
            for key, annotations in folder_annotations.items():
                if annotations:
                    with open(annotation_files[key], "w") as f:
                        json.dump(annotations, f, indent=4)
                    print(f"{key.capitalize()} annotations saved to {annotation_files[key]}")


def main():
    """
    Entry point — update paths below to point at your local dataset copy.

    ``split_data=True`` writes ``all_annotate.json`` from ``parent_folder``.
    Set ``split_data=False`` and fill train/test/val folder paths to emit
    per-split annotation files instead.
    """
    config = {
        "output_folder": "output/annotations",
        "train_folder": "/path/to/dataset/train",
        "test_folder": "/path/to/dataset/test",
        "val_folder": "/path/to/dataset/val",
        "forgery_types": ["duplicate", "delete", "insert", "original"],
        "split_data": True,
    }
    # Root of the video corpus (e.g. balanced VFD / TDTVD / VIFFD merge)
    parent_folder = "/path/to/dataset/merged"
    extractor = VideoFrameExtractor(parent_folder, config)
    extractor.create_annotations()


if __name__ == "__main__":
    main()
