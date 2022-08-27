from pathlib import Path
import typing as tp
import json


def serialize_json(dataset: tp.Dict[str, tp.Dict[str, str]], file_path: Path) -> None:
    """
    Serialize the dataset into JSON file
    :param dataset: dataset structure
    :param file_path: the path where store the file
    :return: None
    """
    json_object = json.dumps(dataset, indent=4)
    with open(file_path, 'w') as fp:
        fp.write(json_object)
