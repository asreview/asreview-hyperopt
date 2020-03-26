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

from asreview.balance_strategies.utils import get_balance_class
from asreview.feature_extraction.utils import get_feature_class
from asreview.models.utils import get_model_class
from asreview import ASReviewData

from asreviewcontrib.hyperopt.job_utils import get_trial_fp
from asreviewcontrib.hyperopt.job_utils import get_split_param
from asreviewcontrib.hyperopt.job_utils import empty_shared
from asreviewcontrib.hyperopt.job_utils import data_fp_from_name
from asreviewcontrib.hyperopt.job_utils import quality
from asreviewcontrib.hyperopt.job_utils import get_out_fp
from asreviewcontrib.hyperopt.job_utils import get_label_fp
from asreviewcontrib.hyperopt.serial_executor import serial_executor


class PassiveJobRunner():
    def __init__(self, data_names, model_name, balance_name, feature_name,
                 executor=serial_executor, n_run=10, server_job=False,
                 data_dir="data", output_dir=None):

        self.trials_dir, self.trials_fp = get_trial_fp(
            data_names, model_name=model_name, balance_name=balance_name,
            feature_name=feature_name, hyper_type="passive",
            output_dir=output_dir)

        self.model_name = model_name
        self.balance_name = balance_name
        self.feature_name = feature_name

        self.model_class = get_model_class(model_name)
        self.feature_class = get_feature_class(feature_name)
        self.balance_class = get_balance_class(balance_name)

        self.server_job = server_job
        self.data_names = data_names
        self.executor = executor
        self.n_run = n_run
        self.data_dir = data_dir
        self._cache = {data_name: {"train_idx": {}}
                       for data_name in data_names}

    def create_loss_function(self):
        def objective_func(param):
            jobs = create_jobs(param, self.data_names, self.n_run)

            self.executor(jobs, self, stop_workers=False,
                          server_job=self.server_job)
            losses = []
            for data_name in self.data_names:
                label_fp = get_label_fp(self.trials_dir, data_name)
                res_files = [get_out_fp(self.trials_dir, data_name, i_run)
                             for i_run in range(self.n_run)]
                losses.append(loss_from_files(res_files, label_fp))
            return {"loss": np.average(losses), 'status': STATUS_OK}

        return objective_func

    def execute(self, param, data_name, i_run):
        split_param = get_split_param(param)
        model = self.model_class(**split_param["model_param"])
        balance_model = self.balance_class(**split_param["balance_param"])
        feature_model = self.feature_class(**split_param["feature_param"])

        as_data = self.get_cached_as_data(data_name)
        train_idx = self.get_cached_train_idx(data_name, i_run)
        out_fp = get_out_fp(self.trials_dir, data_name, i_run)

        np.random.seed(i_run)
        X = feature_model.fit_transform(
            as_data.texts, as_data.title, as_data.abstract)
        X_train, y_train = balance_model.sample(
                X, as_data.labels, train_idx, empty_shared())
        model.fit(X_train, y_train)
        proba = model.predict_proba(X)[:, 1]

        with open(out_fp, "w") as fp:
            json.dump(
                {"proba": proba.tolist(), "train_idx": train_idx.tolist()},
                fp)

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

    def get_cached_train_idx(self, data_name, i_run):
        try:
            return self._cache[data_name]["train_idx"][i_run]
        except KeyError:
            pass

        as_data = self.get_cached_as_data(data_name)
        train_idx = compute_train_idx(as_data.labels, i_run)
        self._cache[data_name]["train_idx"][i_run] = train_idx
        return train_idx

    def get_hyper_space(self):
        model_hs, model_hc = self.model_class().hyper_space()
        balance_hs, balance_hc = self.balance_class().hyper_space()
        feature_hs, feature_hc = self.feature_class().hyper_space()
        hyper_space = {**model_hs, **balance_hs, **feature_hs}
        hyper_choices = {**model_hc, **balance_hc, **feature_hc}
        return hyper_space, hyper_choices

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
                "model_name": self.model_name,
                "balance_name": self.balance_name,
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
    results = {}
    for data_fp in data_fps:
        with open(data_fp, "r") as fp:
            data = json.load(fp)
        train_idx = np.array(data["train_idx"])
        proba = np.array(data["proba"])
        test_idx = np.delete(np.arange(len(labels)), train_idx)
        proba_test = [
            (idx, -proba[idx]) for idx in test_idx]
        proba_test = sorted(proba_test, key=lambda x: x[1])
        for position, item in enumerate(proba_test):
            idx = item[0]
            if labels[idx] == 1:
                if idx not in results:
                    results[idx] = [0, 0]
                results[idx][0] += position
                results[idx][1] += 1

    result_list = []
    for key, item in results.items():
        new_value = item[0]/(item[1]*(len(labels)-len(train_idx)))
        result_list.append([int(key), new_value])

    result_list = sorted(result_list, key=lambda x: x[1])

    return quality(result_list, 1.0)


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


def compute_train_idx(y, seed):
    np.random.seed(seed)
    one_idx = np.where(y == 1)[0]
    zero_idx = np.where(y == 0)[0]

    n_zero_train = min(len(zero_idx)-1, max(1, round(0.75*len(zero_idx))))
    n_one_train = min(len(one_idx)-1, max(1, round(0.75*len(one_idx))))

    train_one_idx = np.random.choice(one_idx, n_one_train, replace=False)
    train_zero_idx = np.random.choice(zero_idx, n_zero_train, replace=False)
    train_idx = np.append(train_one_idx, train_zero_idx)
    return train_idx
