import typing as tp
from pathlib import Path
import logging as lg


def _load_single_dataset(file_path: Path):
    """
    This function read a TSV dataset and returns a list of couple, where the first element is the source language
    and second is the target language. For example [("hi", "ciao"), ("pizza", "pizza")].
    :param file_path: The path of dataset
    :return: List of couple
    """
    # read the TSV file line per line. Each line is a couple (str, str), encoded with "[str]\tab[str]"
    dataset: tp.List[tp.Tuple[str, str]] = []
    with open(file_path, newline='\n', encoding="utf-8") as fp:
        # iterate all lines
        for line in fp:
            # remove the trailing \n
            line = line.replace("\n", "")
            # split using tab delimiter
            data = line.split(sep='\t')
            # append into dataset
            dataset.append((data[0], data[1]))

    return dataset


def _load_datasets(dir_path: Path, source_lang: str) -> tp.Tuple[
    tp.Dict[str, tp.List[tp.Tuple[str, str]]], tp.List[str]]:
    """Load datasets from path and returns a tuple """

    # create dictionary
    datasets = {}
    targets = []

    for file in dir_path.iterdir():
        if file.suffix != '.tsv':
            lg.warning(f"Skipping {file}, it's not tsv file.")
            continue

        lg.info(f"Reading file {file}")

        # getting the dataset info from filename
        # We assume that each file name is formatted as [dataset-name]-[src]-[target]-[stage].tsv
        # remove extension
        file_name = file.name.replace(".tsv", "")
        # split using dash
        file_name_split = file_name.split("-")
        # check the source
        if file_name_split[1] != source_lang:
            lg.error(f"Source language '{file_name_split[1]}' for this file is not supported., skipping file")
            continue

        # get target
        target = file_name_split[2]
        targets.append(target)

        # load dataset
        dataset = _load_single_dataset(file)

        if target in datasets:
            datasets[target] += dataset
        else:
            datasets[target] = dataset

    return datasets, targets


def build_dataset(dir_path: Path, source_lang: str) -> tp.Tuple[tp.Dict[str, tp.Dict[str, str]], tp.List[str]]:
    """
    Build the dataset.
    :param dir_path: directory where all datasets are stored
    :param source_lang: the source language
    :return: a dictionary with this structure
        { "source_sentence1" :
            {
                "target_lang1": "target_sentence1",
                "target_lang2": "target_sentence1",
            },
            ...
        }
    """
    datasets, languages = _load_datasets(dir_path, source_lang)
    dataset = {}
    for target, ds in datasets.items():
        for d in ds:
            key = d[0]
            if key not in dataset:
                dataset[key] = {}
            dataset[key][target] = d[1]

    # for each key search in the loaded dataset the target sentence
    return dataset, languages


def clean_dataset(dataset: tp.Dict[str, tp.Dict[str, str]], target_languages: tp.List[str]) -> tp.Dict[
    str, tp.Dict[str, str]]:
    """
    It cleans the dataset from not-translated keys.
    :param dataset: the dataset to be cleaned
    :param target_languages: the target languages that the dataset must have.
    :return: The cleaned dataset
    """
    keys_to_delete = []

    for key, d in dataset.items():
        parallel = all(target in d for target in target_languages)
        if not parallel:
            keys_to_delete.append(key)

    for k in keys_to_delete:
        del dataset[k]

    return dataset
