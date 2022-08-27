from preprocessing import build_dataset, clean_dataset, serialize_json, compute_metrics, serialize_metrics
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
# Build the dataset
dataset, target_languages = build_dataset(Path(__file__).parent / "datasets/train", "en")
# calculate pre-clean metrics
pre_clean_metrics = compute_metrics((dataset, target_languages))
# serialize the metrics
serialize_metrics(pre_clean_metrics, Path(__file__).parent / "processed/pre_clean_metrics.json")
# clean the dataset
dataset = clean_dataset(dataset, target_languages)
# compute post clean metrics
post_clean_metrics = compute_metrics((dataset, target_languages))
# serialize the post clean metrics
serialize_metrics(post_clean_metrics, Path(__file__).parent / "processed/post_clean_metrics.json")

# save the dataset
serialize_json(dataset, file_path=Path(__file__).parent / "processed/parallel.json")
