import json
import math
import typing as tp
from functools import reduce
from pathlib import Path
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

N_DEGREE = os.cpu_count()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def _calculate_num_sentences(dataset: tp.Dict[str, tp.Dict[str, str]]) -> int:
    sentences = len(dataset.keys())

    for key, targets in dataset.items():
        sentences += len(targets)

    return sentences


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

    def compute_chunk(ds):
        tokens = 0
        tot_len = 0
        for en, value in tqdm(ds):
            tokens += len(tokenizer.tokenize(en))
            tot_len += 1
            for lan in value.values():
                tokens += len(tokenizer.tokenize(lan))
                tot_len += 1
        return tokens, tot_len

    with ThreadPoolExecutor() as executor:
        chunk_len = math.ceil(len(dataset) / N_DEGREE)
        chunks_couples = [(i * chunk_len, min(len(dataset) - 1, (i + 1) * chunk_len - 1)) for i in range(N_DEGREE)]
        futures = [executor.submit(compute_chunk, list(dataset.items())[start:end]) for start, end in chunks_couples]
        tokens, tot_len = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), [future.result() for future in futures])

    metrics['total_tokens'] = tokens
    metrics['avg_tokens_per_sentence'] = tokens / tot_len
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
    metrics['num_sentences'] = _calculate_num_sentences(dataset)
    metrics = {**metrics, **_calculate_parallelization_metrics(dataset, targets_langs)}
    metrics = {**metrics, **_calculate_tokens_metrics(dataset)}

    return metrics


def serialize_metrics(metrics: tp.Dict[str, tp.Any], file_path: Path) -> None:
    with open(file_path, 'w') as fp:
        dump = json.dumps(metrics, indent=4)
        fp.write(dump)
