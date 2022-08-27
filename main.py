from preprocessing import build_dataset, clean_dataset, serialize_json
from pathlib import Path
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

"""
The structure of dataset is the following:
{
    "sentence1": {
        "lang1": "target_lang1",
        "lang2": "target_lang2",
    },
    "sentence2": {
        "lang1": "target_lang1"
        "lang2!: "target_lang2",
    }
}
"""

dataset, target_languages = build_dataset(Path(__file__).parent / "datasets/train", "en")
dataset = clean_dataset(dataset, target_languages)
serialize_json(dataset, file_path=Path(__file__).parent / "processed/parallel.json")
