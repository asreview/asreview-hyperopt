import os
from os.path import join
from pytest import mark
from pathlib import Path

import numpy as np

from asreviewcontrib.hyperopt.cluster import main
from asreviewcontrib.hyperopt.show_trials import load_trials


def remove_dir(output_dir):
    files = [
        join(output_dir, "best", "embase_labelled", "labels.json"),
        join(output_dir, "best", "embase_labelled", "results_0.json"),
        join(output_dir, "best", "embase_labelled", "results_1.json"),
        join(output_dir, "current", "embase_labelled", "labels.json"),
        join(output_dir, "current", "embase_labelled", "results_0.json"),
        join(output_dir, "current", "embase_labelled", "results_1.json"),
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
    "feature_extraction",
    [
        "tfidf",
        "doc2vec",
    ]
)
def test_passive(request, feature_extraction):
    test_dir = request.fspath.dirname
    data_dir = Path(test_dir, "data")
    base_output_dir = Path(test_dir, "temp")
    output_dir = os.path.join(str(base_output_dir),
                              f"fex_{feature_extraction}")
    args = ["--feature_extraction", feature_extraction,
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
