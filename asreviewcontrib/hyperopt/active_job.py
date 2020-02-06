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
import pickle
from distutils.dir_util import copy_tree

from hyperopt import STATUS_OK, Trials, fmin, tpe
import numpy as np
from tqdm import tqdm


from asreview.analysis.analysis import Analysis
from asreview.balance_strategies.utils import get_balance_model
from asreview.feature_extraction.utils import get_feature_model
from asreview.models.utils import get_model
from asreview.query_strategies.utils import get_query_model
from asreview.readers import ASReviewData
from asreview.review.factory import get_reviewer

from asreviewcontrib.hyperopt.job_utils import get_trial_fp
from asreviewcontrib.hyperopt.job_utils import get_split_param
from asreviewcontrib.hyperopt.job_utils import data_fp_from_name
from asreviewcontrib.hyperopt.serial_executor import serial_executor


class ActiveJobRunner():
    def __init__(self, data_names, model_name, query_name, balance_name,
                 feature_name, executor=serial_executor,
                 n_run=8, n_papers=1502, n_instances=50, n_included=1,
                 n_excluded=1):

        self.trials_dir, self.trials_fp = get_trial_fp(
            data_names, model_name=model_name, balance_name=balance_name,
            query_name=query_name, feature_name=feature_name,
            hyper_type="active")

        self.feature_name = feature_name
        self.balance_name = balance_name
        self.query_name = query_name
        self.model_name = model_name

        self.data_names = data_names
        self.executor = executor
        self.n_run = n_run
        self.n_papers = n_papers
        self.n_instances = n_instances
        self.n_included = n_included
        self.n_excluded = n_excluded

        self.data_dir = "data"
        self._cache = {data_name: {"priors": {}}
                       for data_name in data_names}

    def create_loss_function(self):
        def objective_func(param):
            jobs = create_jobs(param, self.data_names, self.n_run)

            self.executor(jobs, self, stop_workers=False)
            losses = []
            for data_name in self.data_names:
                data_dir = os.path.join(self.trials_dir, 'current', data_name)
                losses.append(loss_from_dir(data_dir))
            return {"loss": np.average(losses), 'status': STATUS_OK}

        return objective_func

    def execute(self, param, data_name, i_run):
        split_param = get_split_param(param)
        log_file = get_log_file_name(self.trials_dir, data_name, i_run)
        try:
            os.remove(log_file)
        except FileNotFoundError:
            pass

        prior_included, prior_excluded = self.get_cached_priors(
            data_name, i_run)

        reviewer = get_reviewer(
            data_fp_from_name(self.data_dir, data_name),
            mode='simulate', model=self.model_name,
            query_strategy=self.query_name, balance_strategy=self.balance_name,
            feature_extraction=self.feature_name, n_instances=self.n_instances,
            n_papers=self.n_papers, log_file=log_file,
            prior_included=prior_included, prior_excluded=prior_excluded,
            **split_param)

        reviewer.review()

    def get_cached_priors(self, data_name, i_run):
        try:
            return self._cache[data_name]["priors"][i_run]
        except KeyError:
            pass

        try:
            as_data = self._cache[data_name]["as_data"]
        except KeyError:
            data_fp = data_fp_from_name(self.data_dir, data_name)
            as_data = ASReviewData.from_file(data_fp)
            self._cache[data_name]["as_data"] = as_data

        np.random.seed(i_run)
        ones = np.where(as_data.labels == 1)[0]
        zeros = np.where(as_data.labels == 0)[0]
        included = np.random.choice(ones, self.n_included, replace=False)
        excluded = np.random.choice(zeros, self.n_excluded, replace=False)
        self._cache[data_name]["priors"][i_run] = (included, excluded)

        return included, excluded

    def get_hyper_space(self):
        model_hs, model_hc = get_model(self.model_name).hyper_space()
        query_hs, query_hc = get_query_model(self.query_name).hyper_space()
        balance_hs, balance_hc = get_balance_model(
            self.balance_name).hyper_space()
        feature_hs, feature_hc = get_feature_model(
            self.feature_name).hyper_space()
        hyper_space = {**model_hs, **query_hs, **balance_hs, **feature_hs}
        hyper_choices = {**model_hc, **query_hc, **balance_hc, **feature_hc}
        return hyper_space, hyper_choices

    def hyper_optimize(self, n_iter):
        obj_function = self.create_loss_function()
        hyper_space, hyper_choices = self.get_hyper_space()

        try:
            with open(self.trials_fp, "rb") as fp:
                trials, _ = pickle.load(fp)
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
            with open(self.trials_fp, "wb") as fp:
                pickle.dump((trials, hyper_choices), fp)
            if trials.best_trial['tid'] == len(trials.trials)-1:
                copy_tree(os.path.join(self.trials_dir, "current"),
                          os.path.join(self.trials_dir, "best"))


def loss_spread(time_results, n_papers, moment=1.0):
    loss = 0
    for label in time_results:
        loss += (time_results[label]/n_papers)**moment
    return (loss**(1/moment))/len(time_results)


def loss_from_dir(data_dir):
    analysis = Analysis.from_dir(data_dir)
    results = analysis.avg_time_to_discovery()
    return loss_spread(results, len(analysis.labels))


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


def get_log_file_name(trials_dir, data_name, i_run):
    return os.path.join(trials_dir, "current", data_name,
                        f"results_{i_run}.h5")
