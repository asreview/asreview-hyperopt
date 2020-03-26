## ASReview-hyperopt

![Deploy and release](https://github.com/asreview/asreview-hyperopt/workflows/Deploy%20and%20release/badge.svg)![Build status](https://github.com/asreview/asreview-hyperopt/workflows/test-suite/badge.svg)

Hyper parameter optimization extension for 
[ASReview](https://github.com/asreview/asreview). It uses the 
[hyperopt](https://github.com/hyperopt/hyperopt) package to quickly optimize parameters
of the different models. The hyper parameters and their sample space are defined in the
[ASReview](https://github.com/asreview/asreview) package, and 
automatically used for hyper parameter optimization.

### Installation

The easiest way to install the hyper parameter optimization package is to use the command line:

``` bash
pip install asreview-hyperopt
```

After installation of the visualization package, asreview should automatically detect it.
Test this by:

```bash
asreview --help
```

It should list three new entry points: `hyper-active`, `hyper-passive` and `hyper-cluster`.

### Basic usage

The three entry-points are used in a roughly similar fashion. The main difference between them is
the types of models that have to be supplied:

- hyper-cluster: feature_extraction
- hyper-passive: model, balance\_strategy, feature\_extraction
- hyper-active: model, balance\_strategy, query\_strategy, feature\_extraction


To get help for entry points type:

```bash
asreview hyper-active --help
```

Which results in the following options:

```bash
usage: hyper-active [-h] [-n N_ITER] [-r N_RUN] [-d DATASETS] [--mpi]
                    [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
                    [--server_job] [-m MODEL] [-q QUERY_STRATEGY]
                    [-b BALANCE_STRATEGY] [-e FEATURE_EXTRACTION]

optional arguments:
  -h, --help            show this help message and exit
  -n N_ITER, --n_iter N_ITER
                        Number of iterations of Bayesian Optimization.
  -r N_RUN, --n_run N_RUN
                        Number of runs per dataset.
  -d DATASETS, --datasets DATASETS
                        Datasets to use in the hyper parameter optimization
                        Separate by commas to use multiple at the same time
                        [default: all].
  --mpi                 Use the mpi implementation.
  --data_dir DATA_DIR   Base directory with data files.
  --output_dir OUTPUT_DIR
                        Output directory for trials.
  --server_job          Run job on the server. It will incur less overhead of
                        used CPUs, but more latency of workers waiting for the
                        server to finish its own job. Only makes sense in
                        combination with the flag --mpi.
  -m MODEL, --model MODEL
                        Prediction model for active learning.
  -q QUERY_STRATEGY, --query_strategy QUERY_STRATEGY
                        Query strategy for active learning.
  -b BALANCE_STRATEGY, --balance_strategy BALANCE_STRATEGY
                        Balance strategy for active learning.
  -e FEATURE_EXTRACTION, --feature_extraction FEATURE_EXTRACTION
                        Feature extraction method.

```

### Data structure

The extension will by default search for datasets in the `data` directory, relative to the current
working directory. Either put your datasets there, or specify and data directory.

The output of the runs will by default be stored in the `output` directory, relative to
the current path.

An example of a structure that has been created:

```bash
output/
├── active_learning
│   ├── nb_max_double_tfidf
│   │   └── depression_hall_ace_ptsd_nagtegaal
│   │       ├── best
│   │       │   ├── ace
│   │       │   ├── depression
│   │       │   ├── hall
│   │       │   ├── nagtegaal
│   │       │   └── ptsd
│   │       ├── current
│   │       │   ├── ace
│   │       │   ├── depression
│   │       │   ├── hall
│   │       │   ├── nagtegaal
│   │       │   └── ptsd
│   │       └── trials.pkl
│   └── nb_max_random_double_tfidf
│       └── nagtegaal
│           ├── best
│           │   └── nagtegaal
│           ├── current
│           │   └── nagtegaal
│           └── trials.pkl
├── cluster
│   └── doc2vec
│       ├── ace
│       │   ├── best
│       │   │   └── ace
│       │   ├── current
│       │   │   └── ace
│       │   └── trials.pkl
│       ├── depression_hall_ace_ptsd_nagtegaal
│       │   └── current
│       │       ├── ace
│       │       ├── depression
│       │       ├── hall
│       │       ├── nagtegaal
│       │       └── ptsd
│       └── nagtegaal
│           └── current
│               └── nagtegaal
└── passive
    └── nb_double_tfidf
        └── depression
            ├── best
            │   └── depression
            ├── current
            │   └── depression
            └── trials.pkl
```

The files with name `trials.pkl` are special files that contain data on which trials were run.

To list these trials, use the following command:

```bash
asreview show $SOME_DIRECTORY/trials.pkl
```

It should give a list of trials sorted by the loss (lower is better). The column names (apart
from the loss) are prefixed with the kind of parameter it is:

- `mdl`: Model parameter
- `bal`: Balance strategy parameter
- `qry`: Query strategy parameter
- `fex`: Feature extraction parameter

### Options

The default number of iterations is 1, which you'll probably want to increase. It depends on the
number of hyper-parameters that need to be optimized, but several hundred iterations is probably
a good estimate for most combinations to get reasonably close to the optimum. In all cases,
use good common sense; if the loss is still going down at a quick pace, do a few more iterations.

The hyperopt extension has built-in support for MPI. MPI is used for parallelization of runs. On
a local PC with an MPI-implementation (like OpenMPI) installed, one could run with 4 cores:

```bash
mpirun -n 4 asreview hyper-active --mpi
```

If you want to be slightly more efficient on a machine with a low number of cores, you can run
jobs on the MPI server as well:

```bash
mpirun -n 4 asreview hyper-active --mpi --server_job
```

On super computers one should sometimes replace `mpirun` with `srun`.
