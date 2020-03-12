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

from mpi4py import MPI


def mpi_worker(job_runner):
    comm = MPI.COMM_WORLD
    while True:
        job = comm.recv(source=0)
        if job is None:
            break

        job_runner.execute(**job)
        comm.send(None, dest=0)
    return None, None


def mpi_executor(all_jobs, job_runner=None, server_job=False,
                 stop_workers=True):
    comm = MPI.COMM_WORLD
    n_proc = comm.Get_size()

    for i_proc in range(1, n_proc):
        try:
            job = all_jobs.pop()
        except IndexError:
            break

        comm.send(job, dest=i_proc)

    if server_job:
        try:
            job = all_jobs.pop()
            job_runner.execute(**job)
        except IndexError:
            pass

    n_jobs_sent = 0
    while len(all_jobs) > 0:
        job = all_jobs.pop()
        if server_job and (n_jobs_sent % n_proc) == n_proc - 1:
            job_runner.execute(**job)
            n_jobs_sent += 1
            continue

        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        pid = status.source
        comm.send(job, dest=pid)
        n_jobs_sent += 1

    for i_proc in range(1, n_proc):
        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        pid = status.source
        if stop_workers:
            comm.send(None, dest=pid)


def mpi_hyper_optimize(job_runner, n_iter):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        job_runner.hyper_optimize(n_iter)
        for pid in range(1, comm.Get_size()):
            comm.send(None, dest=pid)
    else:
        mpi_worker(job_runner)
