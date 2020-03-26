# Copyright 2020 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os.path import isfile
import json
import pickle
from distutils.dir_util import copy_tree

from hyperopt import STATUS_OK, Trials, fmin, tpe
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from asreview import ASReviewData
from asreview.feature_extraction.utils import get_feature_class

from asreviewcontrib.hyperopt.cluster_utils import normalized_cluster_score
from asreviewcontrib.hyperopt.job_utils import get_trial_fp
from asreviewcontrib.hyperopt.job_utils import get_split_param
from asreviewcontrib.hyperopt.job_utils import data_fp_from_name
from asreviewcontrib.hyperopt.job_utils import get_label_fp
from asreviewcontrib.hyperopt.job_utils import get_out_fp
from asreviewcontrib.hyperopt.serial_executor import serial_executor


class ClusterJobRunner():
    def __init__(self, data_names, feature_name, executor=serial_executor,
                 n_cluster_run=30, n_feature_run=1, server_job=False,
                 data_dir="data", output_dir=None):

        self.trials_dir, self.trials_fp = get_trial_fp(
            data_names, feature_name=feature_name, hyper_type="cluster",
            output_dir=output_dir)

        self.feature_name = feature_name
        self.feature_class = get_feature_class(feature_name)

        self.data_names = data_names
        self.executor = executor
        self.n_cluster_run = n_cluster_run
        self.n_feature_run = n_feature_run
        self.data_dir = data_dir
        self.server_job = server_job
        self._cache = {data_name: {}
                       for data_name in data_names}

    def create_loss_function(self):
        def objective_func(param):
            jobs = create_jobs(param, self.data_names, self.n_feature_run)

            self.executor(jobs, self, stop_workers=False,
                          server_job=self.server_job)
            losses = []
            for data_name in self.data_names:
                label_fp = get_label_fp(self.trials_dir, data_name)
                res_files = [get_out_fp(self.trials_dir, data_name, i_run)
                             for i_run in range(self.n_feature_run)]
                losses.append(loss_from_files(res_files, label_fp))
            return {"loss": np.average(losses), 'status': STATUS_OK}

        return objective_func

    def execute(self, param, data_name, i_run):
        split_param = get_split_param(param)
        feature_model = self.feature_class(**split_param["feature_param"])

        as_data = self.get_cached_as_data(data_name)
        out_fp = get_out_fp(self.trials_dir, data_name, i_run)

        X = feature_model.fit_transform(
            as_data.texts, as_data.title, as_data.abstract)

        n_clusters = max(2, int(len(as_data.labels)/200))
        np.random.seed(i_run)
        all_predictions = []
        for _ in range(self.n_cluster_run):
            kmeans_model = KMeans(n_clusters=n_clusters, n_init=1, n_jobs=1)
            all_predictions.append(kmeans_model.fit_predict(X).tolist())

        with open(out_fp, "w") as fp:
            json.dump({"predictions": all_predictions}, fp)

        label_fp = get_label_fp(self.trials_dir, data_name)
        if i_run == 0 and not isfile(label_fp):
            with open(label_fp, "w") as fp:
                json.dump(as_data.labels.tolist(), fp)

    def get_cached_as_data(self, data_name):
        try:
            return self._cache[data_name]["as_data"]
        except KeyError:
            pass
        data_fp = data_fp_from_name(self.data_dir, data_name)
        as_data = ASReviewData.from_file(data_fp)
        self._cache[data_name]["as_data"] = as_data
        return as_data

    def get_hyper_space(self):
        return self.feature_class().hyper_space()

    def hyper_optimize(self, n_iter):
        obj_function = self.create_loss_function()
        hyper_space, hyper_choices = self.get_hyper_space()

        try:
            with open(self.trials_fp, "rb") as fp:
                trials_data = pickle.load(fp)
            trials = trials_data["trials"]
        except FileNotFoundError:
            trials = None
            print(f"Creating new hyper parameter optimization run: "
                  f"{self.trials_fp}")

        if trials is None:
            trials = Trials()
            n_start_evals = 0
        else:
            n_start_evals = len(trials.trials)

        for i in tqdm(range(n_iter)):
            fmin(fn=obj_function,
                 space=hyper_space,
                 algo=tpe.suggest,
                 max_evals=i+n_start_evals+1,
                 trials=trials,
                 show_progressbar=False)
            trials_data = {
                "trials": trials,
                "hyper_choices": hyper_choices,
                "feature_name": self.feature_name,
            }
            with open(self.trials_fp, "wb") as fp:
                pickle.dump(trials_data, fp)
            if trials.best_trial['tid'] == len(trials.trials)-1:
                copy_tree(os.path.join(self.trials_dir, "current"),
                          os.path.join(self.trials_dir, "best"))


def loss_from_files(data_fps, labels_fp):
    with open(labels_fp, "r") as fp:
        labels = np.array(json.load(fp), dtype=int)

    all_scores = []
    for data_fp in data_fps:
        cur_scores = []

        with open(data_fp, "r") as fp:
            predictions = np.array(json.load(fp)["predictions"], dtype=int)

        for prediction in predictions:
            score = normalized_cluster_score(prediction, labels)
            cur_scores.append(score)
        all_scores.append(cur_scores)

    print(all_scores)
    return -np.average(all_scores)


def create_jobs(param, data_names, n_run):
    jobs = []
    for data_name in data_names:
        for i_run in range(n_run):
            jobs.append({
                "param": param,
                "data_name": data_name,
                "i_run": i_run,
                })
    return jobs
