#!/usr/bin/env bash

export LD_LIBRARY_PATH=/opt/intel/oneapi/redist/opt/mpi/libfabric/lib:$LD_LIBRARY_PATH
export PATH=/opt/intel/oneapi/redist/bin:$PATH
export I_MPI_ROOT=/opt/intel/oneapi/redist/lib
export CCL_ROOT=/opt/intel/oneapi/redist
export FI_PROVIDER_PATH=/opt/intel/oneapi/redist/opt/mpi/libfabric/lib/prov
