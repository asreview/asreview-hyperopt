import os
from os.path import join
from pytest import mark
from pathlib import Path

import numpy as np

from asreviewcontrib.hyperopt.active import main
from asreviewcontrib.hyperopt.show_trials import load_trials

data_dir = Path("tests", "data")
base_output_dir = Path("tests", "temp")


def remove_dir(output_dir):
    files = [
        join(output_dir, "best", "embase_labelled", "results_0.h5"),
        join(output_dir, "best", "embase_labelled", "results_1.h5"),
        join(output_dir, "current", "embase_labelled", "results_0.h5"),
        join(output_dir, "current", "embase_labelled", "results_1.h5"),
        join(output_dir, "trials.pkl")
    ]
    dirs = [
        join(output_dir, "best", "embase_labelled"),
        join(output_dir, "current", "embase_labelled"),
        join(output_dir, "best"),
        join(output_dir, "current"),
        output_dir,
    ]

    for file_ in files:
        try:
            os.remove(file_)
        except FileNotFoundError:
            pass
    for dir_ in dirs:
        try:
            os.rmdir(dir_)
        except (FileNotFoundError, OSError):
            pass


@mark.parametrize(
    "model,feature_extraction,query_strategy,balance_strategy",
    [
        ("nb", "tfidf", "max", "simple"),
        ("rf", "doc2vec", "max_random", "double"),
        ("logistic", "doc2vec", "cluster", "triple"),
        ("nn-2-layer", "doc2vec", "uncertainty", "undersample"),
        ("svm", "tfidf", "uncertainty_max", "simple"),
    ]
)
def test_active(model, feature_extraction, query_strategy, balance_strategy):
    output_dir = os.path.join(str(base_output_dir), model)
    args = ["--model", model,
            "--feature_extraction", feature_extraction,
            "--query_strategy", query_strategy,
            "--balance_strategy", balance_strategy,
            "--data_dir", str(data_dir),
            "--n_run", "2",
            "--output_dir", output_dir,
            "--n_iter", "2"
            ]
    remove_dir(output_dir)
    main(args)
    trial_vals = load_trials(join(output_dir, "trials.pkl"))["values"]
    assert np.all(np.array([len(x) for x in trial_vals.values()]) == 2)
    remove_dir(output_dir)
