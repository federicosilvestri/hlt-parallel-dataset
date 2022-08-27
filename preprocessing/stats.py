import json
import typing as tp
from pathlib import Path

def _calculate_couple_size(dataset: tp.Dict[str, tp.Dict[str, str]]) -> int:
    couple = 0

    for key, targets in dataset.items():
        couple += len(targets)

    return couple


def _calculate_parallelization_metrics(dataset: tp.Dict[str, tp.Dict[str, str]], target_langs: tp.List[str]) -> tp.Dict[
    str, tp.Any]:
    metrics: tp.Dict[str, tp.Any] = dict()

    # non translated records
    ntr = 0
    for key, targets in dataset.items():
        if not all([target in targets for target in target_langs]):
            ntr += 1

    metrics['parallelization_missing_records'] = ntr
    metrics['parallelization_perc'] = (1 - (ntr / len(dataset))) * 100.0

    return metrics


def _calculate_tokens_metrics(dataset: tp.Dict[str, tp.Dict[str, str]]) -> tp.Dict[str, tp.Any]:
    metrics: tp.Dict[str, tp.Any] = dict()
    tokens = 0
    for key, _ in dataset.items():
        tokens += len(key.split())

    metrics['total_tokens'] = tokens
    metrics['avg_tokens_per_sentence'] = tokens / len(dataset)
    return metrics


def compute_metrics(build_output: tp.Tuple[tp.Dict[str, tp.Dict[str, str]], tp.List[str]]) -> tp.Dict[str, tp.Any]:
    """
    Computes the metrics for a specific dataset.
    :param build_output: the output of build_dataset function.
    :return: a dictionary of metric_name, metric_value
    """
    dataset, targets_langs = build_output
    metrics: tp.Dict[str, float] = dict()

    # the size of dataset, how many records we have
    metrics['size'] = len(dataset)
    metrics['couple_size'] = _calculate_couple_size(dataset)
    metrics = {**metrics, **_calculate_parallelization_metrics(dataset, targets_langs)}
    metrics = {**metrics, **_calculate_tokens_metrics(dataset)}

    return metrics


def serialize_metrics(metrics: tp.Dict[str, tp.Any], file_path: Path) -> None:
    with open(file_path, 'w') as fp:
        dump = json.dumps(metrics, indent=4)
        fp.write(dump)