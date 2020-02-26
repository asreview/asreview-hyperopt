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

from configparser import ConfigParser
import sys

import numpy as np

from asreview.entry_points.base import BaseEntryPoint
import argparse
from asreviewcontrib.hyperopt.show_trials import load_trials


PREFIX_VALUES = {
    "fex": "feature_param",
    "bal": "balance_param",
    "mdl": "model_param",
    "qry": "query_param"
}


DEFAULT_CONFIG_GLOBALS = {
    "n_instances": 20,
    "n_prior_included": 1,
    "n_prior_excluded": 1,
}


class CreateConfigEntryPoint(BaseEntryPoint):
    description = "Create a configuration file from a trials file."

    def __init__(self):
        super(CreateConfigEntryPoint, self).__init__()
        from asreviewcontrib.hyperopt.__init__ import __version__
        from asreviewcontrib.hyperopt.__init__ import __extension_name__

        self.extension_name = __extension_name__
        self.version = __version__

    def execute(self, argv):
        parser = _parse_arguments()
        args = vars(parser.parse_args(argv))
        trials_fp = args["trials_fp"]
        trials_data = load_trials(trials_fp)

        values = trials_data["values"]
        with_config = args["with_config"]
        output = args["output"]

        config = ConfigParser()
        if with_config is not None:
            config.read(with_config)
        else:
            config["global_settings"] = DEFAULT_CONFIG_GLOBALS
        min_idx = np.argmin(values["loss"])

        if "global_settings" not in config:
            config["global_settings"] = {}
        global_set = config["global_settings"]

        if "model_name" in trials_data:
            global_set["model"] = trials_data["model_name"]
        if "balance_name" in trials_data:
            global_set["balance_strategy"] = trials_data["balance_name"]
        if "query_name" in trials_data:
            global_set["query_strategy"] = trials_data["query_name"]
        if "feature_name" in trials_data:
            global_set["feature_extraction"] = trials_data["feature_name"]

        for key in values:
            if key[:3] not in PREFIX_VALUES:
                continue
            section = PREFIX_VALUES[key[:3]]
            if section not in config:
                config[section] = {}
            config[section][key[4:]] = str(values[key][min_idx])

        with open(output, "w") as fp:
            config.write(fp)


def _parse_arguments():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument(
        "trials_fp",
        type=str,
        help="Trials file to create configuration file."
    )
    parser.add_argument(
        "--with_config",
        type=str,
        default=None,
        help="Use a configuration file template to set simulation parameters.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="/dev/stdout",
        help="Output of the configuration file."
    )
    return parser
