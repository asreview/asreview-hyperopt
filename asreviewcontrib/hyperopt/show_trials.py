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

from copy import deepcopy
import pickle

import pandas as pd
import numpy as np

from asreview.entry_points.base import BaseEntryPoint


def load_trials(trials_fp):
    with open(trials_fp, "rb") as fp:
        trials_data = pickle.load(fp)

    trials = trials_data["trials"]
    hyper_choices = trials_data["hyper_choices"]

    values = deepcopy(trials.vals)
    for key in values:
        if key in hyper_choices:
            for i in range(len(values[key])):
                values[key][i] = hyper_choices[key][values[key][i]]

    values.update({"loss": trials.losses()})

    for key, arr in values.items():
        if not isinstance(arr[0], float):
            continue
        if np.all(arr-np.array(arr, dtype=int) == 0.0):
            values[key] = [int(val) for val in arr]

    trials_data["values"] = values
    return trials_data


class ShowTrialsEntryPoint(BaseEntryPoint):
    description = "List trials for hyper parameter optimization."

    def __init__(self):
        super(ShowTrialsEntryPoint, self).__init__()
        from asreviewcontrib.hyperopt.__init__ import __version__
        from asreviewcontrib.hyperopt.__init__ import __extension_name__

        self.extension_name = __extension_name__
        self.version = __version__

    def execute(self, argv):
        try:
            trials_fp = argv[0]
        except IndexError:
            print("Error: need argument path to trials.pkl file.")
        values = load_trials(trials_fp)["values"]
        pd.options.display.max_rows = 999
        pd.options.display.width = 0
        print(pd.DataFrame(values).sort_values("loss"))
