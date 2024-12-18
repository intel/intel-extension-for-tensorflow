#!/usr/bin/env bash
# Copyright (c) 2021-2022 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

onednn_gpu_path=$(find bazel-out/k8-opt-ST-*/bin/external/onednn_gpu/include/oneapi/dnnl/ -name dnnl_version.h | head -1)
onednn_cpu_path=$(find bazel-out/k8-opt-ST-*/bin/external/onednn_cpu/include/oneapi/dnnl/ -name dnnl_version.h | head -1)

itex_tmp_folder_name="itex.tmp"
lib_tmp_folder_name="lib.tmp"

set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function get_git_desc() {
  git_version=`git rev-parse --short=8 HEAD`
  echo $git_version
}

function get_compiler_version() {
  compiler_path=`cat .itex_configure.bazelrc | grep -Eo 'DPCPP_TOOLKIT_PATH=.*$' | cut -d '=' -f 2 | cut -d '"' -f 2`
  version=`${compiler_path}/bin/icx --version | grep -Eo '\([a-zA-Z0-9.]{10,}\)' | grep -Eo '[a-zA-Z0-9.]{10,}'`
  echo "dpcpp-${version}"
}

function get_onednn_git_version() {
  onednn_path=$1
  if [ ! -f ${onednn_path} ]; then
    echo "none"
  else
    major_version=`cat ${onednn_path} | grep '#define DNNL_VERSION_MAJOR' | cut -d ' ' -f 3`
    minor_version=`cat ${onednn_path} | grep '#define DNNL_VERSION_MINOR' | cut -d ' ' -f 3`
    patch_version=`cat ${onednn_path} | grep '#define DNNL_VERSION_PATCH' | cut -d ' ' -f 3`
    commit=`cat ${onednn_path} | grep '#define DNNL_VERSION_HASH' | grep -Eo '[a-z0-9]{40}'`
    version="v${major_version}.${minor_version}.${patch_version}-`echo ${commit} | cut -c 1-8`"
    echo $version
  fi
}

function emit_version_info() {
  if [ ! -f $1 ]; then
    echo "$1 not exists!"
    exit -1
  fi

  if [ ! -f $2 ]; then
    echo "$2 not exists!"
    exit -1
  fi

  itex_version=$(grep -E "^__version__ *= *'[0-9]+.[0-9]+.[0-9]+.*'" $2 | awk -F"'" '{print $2}')
  # weekly build, add .dev suffix
  if [ ! -z "$3" ]; then
    itex_version="${itex_version}.dev$(date +%Y%m%d)"
  fi
  echo "__version__ = '$itex_version'" >> $1
  echo "__git_desc__= '`get_git_desc`'" >> $1
  echo "VERSION = __version__" >> $1
  echo "GIT_VERSION = 'v' + __version__ + '-' + __git_desc__" >> $1
  echo "COMPILER_VERSION = '`get_compiler_version`'" >> $1
  onednn_version_check=0
  if [ ! -z "${onednn_gpu_path}" ] && [ -f "${onednn_gpu_path}" ]; then
    onednn_gpu_git_version=`get_onednn_git_version ${onednn_gpu_path}`
    if [ ${onednn_gpu_git_version} != "none" ]; then
      echo "ONEDNN_GPU_GIT_VERSION = '${onednn_gpu_git_version}'" >> $1
    fi
    onednn_version_check=$(($onednn_version_check + 1))
  fi
  if [ ! -z "${onednn_cpu_path}" ] && [ -f "${onednn_cpu_path}" ]; then
    onednn_cpu_git_version=`get_onednn_git_version ${onednn_cpu_path}`
    if [ ${onednn_cpu_git_version} != "none" ]; then
      echo "ONEDNN_CPU_GIT_VERSION = '${onednn_cpu_git_version}'" >> $1
    fi
    onednn_version_check=$(($onednn_version_check + 1))
  fi
  if [ $onednn_version_check -eq 0 ]; then
    echo "Error: no oneDNN version files"
    exit -1
  fi
  echo "TF_COMPATIBLE_VERSION = '>= 2.8.0'" >> $1
}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function prepare_src() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  TMPDIR="$1"
  mkdir -p "$TMPDIR"
  ITEX_TMPDIR="$TMPDIR/$itex_tmp_folder_name"
  LIB_TMPDIR="$TMPDIR/$lib_tmp_folder_name"
  mkdir -p "$ITEX_TMPDIR"
  mkdir -p "$LIB_TMPDIR"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"
  echo $(date) : "=== Preparing sources in dir: ${ITEX_TMPDIR}"
  echo $(date) : "=== Preparing sources in dir: ${LIB_TMPDIR}"

  if [ ! -d bazel-bin/itex ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  RUNFILES=bazel-bin/itex/tools/pip_package/build_pip_package.runfiles/intel_extension_for_tensorflow
  cp -LR \
      bazel-bin/itex/tools/pip_package/build_pip_package.runfiles/intel_extension_for_tensorflow/itex \
      "${ITEX_TMPDIR}"
  # Copy oneDNN libs over so they can be loaded at runtime
  so_lib_dir=$(ls $RUNFILES | grep solib) || true
  if [ -n "${so_lib_dir}" ]; then
    onednn_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep onednn) || true
    if [ -n "${onednn_so_dir}" ]; then
      mkdir "${LIB_TMPDIR}/${so_lib_dir}"
      cp -R ${RUNFILES}/${so_lib_dir}/${onednn_so_dir} "${LIB_TMPDIR}/${so_lib_dir}"
    fi
  fi
  
  # itex
  cp itex/tools/pip_package/README.md ${ITEX_TMPDIR}/README.md
  cp LICENSE.txt ${ITEX_TMPDIR}/
  cp -r third-party-programs ${ITEX_TMPDIR}/
  cp itex/tools/pip_package/itex_setup.py ${ITEX_TMPDIR}/setup.py
  mkdir -p ${ITEX_TMPDIR}/intel_extension_for_tensorflow
  mv -f ${ITEX_TMPDIR}/third-party-programs ${ITEX_TMPDIR}/intel_extension_for_tensorflow/
  if [ -d ${ITEX_TMPDIR}/itex ] ; then
    mv -f ${ITEX_TMPDIR}/itex/* ${ITEX_TMPDIR}/intel_extension_for_tensorflow
    cp -rf itex/python/* ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python
    cp -rf tools/* ${ITEX_TMPDIR}/intel_extension_for_tensorflow/tools
    mv -f ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python/base_init.py ${ITEX_TMPDIR}/intel_extension_for_tensorflow/__init__.py
    mv -f ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python/__main__.py ${ITEX_TMPDIR}/intel_extension_for_tensorflow/__main__.py
    emit_version_info ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python/version.py ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python/gen_itex_version.py "$WEEKLY_BUILD_FLAG"
    chmod +x ${ITEX_TMPDIR}/intel_extension_for_tensorflow/__init__.py
    rm -rf ${ITEX_TMPDIR}/itex
  fi

  # lib_itex
  cp LICENSE.txt ${LIB_TMPDIR}/
  cp -r third-party-programs ${LIB_TMPDIR}/
  cp -r tools/ ${LIB_TMPDIR}/
  cp itex/tools/pip_package/lib_setup.py ${LIB_TMPDIR}/setup.py
  mkdir -p ${LIB_TMPDIR}/intel_extension_for_tensorflow
  mkdir -p ${LIB_TMPDIR}/tensorflow-plugins
  mkdir -p ${LIB_TMPDIR}/intel_extension_for_tensorflow_lib
  touch ${LIB_TMPDIR}/intel_extension_for_tensorflow_lib/__init__.py
  mv -f ${LIB_TMPDIR}/third-party-programs ${LIB_TMPDIR}/intel_extension_for_tensorflow_lib/
  [ -f "${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_cpu.so" ] && mv ${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_cpu.so ${LIB_TMPDIR}/tensorflow-plugins/
  [ -f "${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_gpu.so" ] && mv ${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_gpu.so ${LIB_TMPDIR}/tensorflow-plugins/
  [ -f "${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_cpu_internal_avx2.so" ] && mv ${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_cpu_internal_avx2.so ${LIB_TMPDIR}/intel_extension_for_tensorflow/
  [ -f "${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_cpu_internal_avx512.so" ] && mv ${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_cpu_internal_avx512.so ${LIB_TMPDIR}/intel_extension_for_tensorflow/
  [ -f "${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_gpu_internal.so" ] && mv ${ITEX_TMPDIR}/intel_extension_for_tensorflow/libitex_gpu_internal.so ${LIB_TMPDIR}/intel_extension_for_tensorflow/
  [ -f "bazel-bin/third_party/onednn/libonednn_cpu_so.so" ] && cp -LR bazel-bin/third_party/onednn/libonednn_cpu*.so ${LIB_TMPDIR}/intel_extension_for_tensorflow/
  cp ${RUNFILES}/../../../../core/kernels/libitex_common.so ${LIB_TMPDIR}/intel_extension_for_tensorflow/
  xetla_lib=$(find ${RUNFILES} -name libitex_gpu_xetla.so)
  [ -f "${xetla_lib}" ] && cp ${xetla_lib} ${LIB_TMPDIR}/intel_extension_for_tensorflow/
  xla_so_lib=$(find ${RUNFILES} -iname libintel_xla.so)
  [ -f "${xla_so_lib}" ] && cp ${xla_so_lib} ${LIB_TMPDIR}/intel_extension_for_tensorflow/

  mkdir ${LIB_TMPDIR}/intel_extension_for_tensorflow/python
  cp -f ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python/version.py ${LIB_TMPDIR}/intel_extension_for_tensorflow/python/
  mv ${ITEX_TMPDIR}/intel_extension_for_tensorflow/python/*wrap* ${LIB_TMPDIR}/intel_extension_for_tensorflow/python
  mkdir ${LIB_TMPDIR}/intel_extension_for_tensorflow/tools
  mv ${LIB_TMPDIR}/tools/* ${LIB_TMPDIR}/intel_extension_for_tensorflow/tools
}

function build_wheel() {
  if [ $# -lt 2 ] ; then
    echo "No src and dest dir provided"
    exit 1
  fi

  TMPDIR="$1"
  DEST="$2"
  PKG_NAME_FLAG="$3"
  WEEKLY_BUILD_FLAG="$4"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e tools/python_bin_path.sh ]]; then
    source tools/python_bin_path.sh
  fi

  pushd ${TMPDIR}/${lib_tmp_folder_name} > /dev/null
  rm -f MANIFEST
  echo $(date) : "=== Building Intel® Extension for Tensorflow* library wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${PKG_NAME_FLAG} ${WEEKLY_BUILD_FLAG}>/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel path: ${DEST}"

  pushd ${TMPDIR}/${itex_tmp_folder_name} > /dev/null
  rm -f MANIFEST
  echo $(date) : "=== Building Intel® Extension for Tensorflow* wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${PKG_NAME_FLAG} ${WEEKLY_BUILD_FLAG}>/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel path: ${DEST}"
}

function usage() {
  echo "Usage:"
  echo "$0 [--src srcdir] [--dst dstdir] [options]"
  echo "$0 dstdir [options]"
  echo ""
  echo "    --src                 prepare sources in srcdir"
  echo "                              will use temporary dir if not specified"
  echo ""
  echo "    --dst                 build wheel in dstdir"
  echo "                              if dstdir is not set do not build, only prepare sources"
  echo ""
  exit 1
}

function main() {
  PKG_NAME_FLAG=""
  PROJECT_NAME=""
  NIGHTLY_BUILD=0
  SRCDIR=""
  DSTDIR=""
  CLEANSRC=1
  WEEKLY_BUILD_FLAG=""
  while true; do
    if [[ "$1" == "--help" ]]; then
      usage
      exit 1
    elif [[ "$1" == "--project_name" ]]; then
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
    elif [[ "$1" == "--src" ]]; then
      shift
      SRCDIR="$(real_path $1)"
      CLEANSRC=0
    elif [[ "$1" == "--weekly" ]]; then
      WEEKLY_BUILD_FLAG="--weekly_build"
    elif [[ "$1" == "--dst" ]]; then
      shift
      DSTDIR="$(real_path $1)"
    else
      DSTDIR="$(real_path $1)"
    fi
    shift

    if [[ -z "$1" ]]; then
      break
    fi
  done

  if [[ -z "$DSTDIR" ]] && [[ -z "$SRCDIR" ]]; then
    echo "No destination dir provided"
    usage
    exit 1
  fi

  if [[ -z "$SRCDIR" ]]; then
    # make temp srcdir if none set
    SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  fi

  prepare_src "$SRCDIR"

  if [[ -z "$DSTDIR" ]]; then
      # only want to prepare sources
      exit
  fi

  if [[ -n ${PROJECT_NAME} ]]; then
    PKG_NAME_FLAG="--project_name ${PROJECT_NAME}"
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG" "$WEEKLY_BUILD_FLAG"

  if [[ $CLEANSRC -ne 0 ]]; then
    rm -rf "${TMPDIR}"
  fi
}

main "$@"
