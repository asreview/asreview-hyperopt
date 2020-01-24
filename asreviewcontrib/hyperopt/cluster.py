import sys
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

import argparse

from asreview.entry_points import BaseEntryPoint

from asreviewcontrib.hyperopt.mpi_executor import mpi_executor
from asreviewcontrib.hyperopt.mpi_executor import mpi_hyper_optimize
from asreviewcontrib.hyperopt.serial_executor import serial_executor
from asreviewcontrib.hyperopt.serial_executor import serial_hyper_optimize
from asreviewcontrib.hyperopt.job_utils import get_data_names
from asreviewcontrib.hyperopt.cluster_job import ClusterJobRunner


class HyperClusterEntryPoint(BaseEntryPoint):
    description = "Hyper parameter optimization for clustering."

    def __init__(self):
        super(HyperClusterEntryPoint, self).__init__()
        from asreviewcontrib.hyperopt.__init__ import __version__
        from asreviewcontrib.hyperopt.__init__ import __extension_name__

        self.extension_name = __extension_name__
        self.version = __version__

    def execute(self, argv):
        main(argv)


def _parse_arguments():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument(
        "-e", "--feature_extraction",
        type=str,
        default="doc2vec",
        help="Feature extraction method.")
    parser.add_argument(
        "-n", "--n_iter",
        type=int,
        default=1,
        help="Number of iterations of Bayesian Optimization."
    )
    parser.add_argument(
        "-d", "--datasets",
        type=str,
        default="all",
        help="Datasets to use in the hyper parameter optimization "
        "Separate by commas to use multiple at the same time [default: all].",
    )
    parser.add_argument(
        "--mpi",
        dest='use_mpi',
        action='store_true',
        help="Use the mpi implementation.",
    )
    return parser


def main(argv=sys.argv[1:]):
    parser = _parse_arguments()
    args = vars(parser.parse_args(argv))
    datasets = args["datasets"].split(",")
    feature_name = args["feature_extraction"]
    n_iter = args["n_iter"]
    use_mpi = args["use_mpi"]

    data_names = get_data_names(datasets)
    if use_mpi:
        executor = mpi_executor
    else:
        executor = serial_executor

    job_runner = ClusterJobRunner(data_names, feature_name, executor=executor)

    if use_mpi:
        mpi_hyper_optimize(job_runner, n_iter)
    else:
        serial_hyper_optimize(job_runner, n_iter)
