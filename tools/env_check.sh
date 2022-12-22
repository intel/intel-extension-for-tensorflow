# Copyright (c) 2022 Intel Corporation
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

ubuntu_version_list=(20.04 22.04)
redhat_version_list=(8.5)

min_python_version=7
max_python_version=10

min_tensorflow_version=10

driver_list_for_ubuntu=(
  "intel-level-zero-gpu"
  "intel-opencl-icd"
  "level-zero"
  "level-zero-dev"
  "libdrm-common"
  "libdrm2"
  "libdrm-amdgpu1"
  "libdrm-intel1"
  "libdrm-nouveau2"
  "libdrm-dev"
  "libigc1"
  "libigdfcl1"
  "libigdgmm12"
)

driver_list_for_rhel=(
  "hwdata"
  "intel-igc-core"
  "intel-igc-opencl"
  "intel-gmmlib"
  "intel-opencl"
  "level-zero"
  "level-zero-devel"
  "libdrm"
  "libpciaccess"
  "libpkgconf"
  "pkgconf"
  "pkgconf-m4"
  "pkgconf-pkg-config"
)

oneapi_list=(compiler tbb mkl ccl)

tf_require_list=(
  "absl-py"
  "astunparse"
  "flatbuffers"
  "gast"
  "google-pasta"
  "grpcio"
  "h5py"
  "Keras|keras-nightly"
  "libclang"
  "numpy"
  "opt-einsum"
  "packaging"
  "protobuf"
  "six"
  "termcolor"
  "typing_extensions"
  "tb-nightly|tensorboard"
  "tensorflow-estimator|tf-estimator-nightly"
  "tensorflow-io-gcs-filesystem"
  "wrapt"
)

itex_require_list=(
  "wheel"
  "requests"
)

IS_HELP=0
IS_DETAIL=0
IS_FAILED=0

set -- `getopt -o gd,h -l gpu,detail,help -n Usage  -- "$@"`
while [ -n "$1" ]; do
  case $1 in
    -d|--detail)
      IS_DETAIL=1
      shift 1
      ;;
    -h|--help)
      IS_HELP=1
      shift 1
      ;;
    --) shift; break;;
  esac
done

usage(){
cat << EOM

Usage: ./env_check.sh [--detail]

Mandatory arguments to long options are mandatory for short options too.
  -d, --detail       print tensorflow and itex required python libraries' installed status.

EOM
}

die() {
  echo -e "\033[31m $1. \033[0m"
  echo ""
  echo -e "====================== \033[31m $2 \033[0m ======================="
  echo ""
  exit -1
}

trim() {
  str=""

  if [ $# -gt 0 ]; then
    str="$1"
  fi
  echo "$str" | sed -e 's/^[ \t\r\n]*//g' | sed -e 's/[ \t\r\n]*$//g' |sed 's/"//g'|sed 's/^\s*//g'
}

check_os() {
  echo ""
  echo -e "========================== \033[33m Check OS \033[0m =========================="
  echo ""
  os_name=$(trim $(cat /etc/os-release 2>/dev/null | grep ^ID= | awk -F= '{print $2}'))
  os_version=$(trim $(cat /etc/os-release 2>/dev/null | grep ^VERSION_ID= | awk -F= '{print $2}'))

  if [ "$os_name" = "" ]; then
    os_name=$(trim $(lsb_release -i 2>/dev/null | awk -F: '{print $2}'))
  fi
  if [ ! "$os_name" = "" ]; then
    os_name=$(echo $os_name | tr '[A-Z]' '[a-z]')
  fi

  case "${os_name}" in
      ubuntu)
        if [[ "${ubuntu_version_list[*]}"  =~ "${os_version}" ]]; then
          echo -e "\033[33m OS ${os_name}:${os_version} is Supported. \033[0m"
          echo ""
        else
          die "Intel GPU Driver Does Not Support OS ${os_name}:${os_version} yet" " Check OS Failed"
        fi
        ;;
      rhel)
        if [[ "${redhat_version_list[*]}"  =~ "${os_version}" ]]; then
          echo -e "\033[33m OS ${os_name}:${os_version} is Supported. \033[0m"
          echo ""
        else
          die "Intel GPU Driver Does Not Support OS ${os_name}:${os_version} yet" " Check OS Failed"
        fi
        ;;
      sles|centos)
        die "Intel GPU Driver Does Not Support OS ${os_name}:${os_version} yet" " Check OS Failed"
        ;;
      *)
        die "Unknow OS ${os_name}" " Check OS Failed"
        ;;
  esac
  echo -e "====================== \033[32m Check OS Passed \033[0m ======================="
  echo ""
}

installed_status() {
  case "${os_name}" in
    ubuntu)
      status=$(dpkg -s $1 2>/dev/null |grep Status|awk -F ':' '{print $2}'|grep "install ok installed"|sed 's/^\s*//g')
      version=$(dpkg -s $1 2>/dev/null |grep Version|awk -F ':' '{print $2}'|sed 's/^\s*//g')
      ;;
    rhel)
      status=$(yum info installed $1 2>/dev/null|grep Name|awk -F ':' '{print $2}')
      version=$(yum info installed $1 2>/dev/null|grep Version|awk -F ':' '{print $2}'|sed 's/^\s*//g')
  esac

  if [[ ! -z ${status} ]]; then
    echo -e "\033[33m Intel(R) graphics runtime $1-${version} is installed. \033[0m"
  else
    echo -e "\033[31m Intel(R) graphics runtime $1 is not installed! \033[0m"
  fi
}

check_intel_gpu_driver() {
  echo ""
  echo -e "=================== \033[33m Check Intel GPU Driver \033[0m ==================="
  echo ""

  if [[ "${os_name}" = "ubuntu" ]]; then
    for driver in ${driver_list_for_ubuntu[@]}
    do
      installed_status ${driver}
    done
  elif [[ "${os_name}" = "rhel" ]]; then
    for driver in ${driver_list_for_rhel[@]}
    do
      installed_status ${driver}
    done
  else
    echo -e "=============== \033[31m Check Intel GPU Driver Failed \033[0m ================"
  fi
  echo ""
  echo -e "=============== \033[32m Check Intel GPU Driver Finshed \033[0m ================"
  echo ""
}

installed_status_oneapi() {
  grep $1 ${LOAD_LIBS}|grep "fini" -q 2>&1
  if [ $? -eq 0 ]; then
    echo -e "\033[33m $2 is installed. \033[0m"
  else
    echo -e "\033[31m Can't find $1, $2 is uninstalled or unset relevant environment viriables, such as $3. \033[0m"
    IS_FAILED=1
  fi
}


check_intel_oneapi() {
  echo ""
  echo -e "===================== \033[33m Check Intel OneApi \033[0m ====================="
  echo ""

  LOAD_LIBS=/tmp/loadlibs
  LD_DEBUG=libs ${python_bin_path} -c "import tensorflow"  2>>${LOAD_LIBS}
  for oneapi in ${oneapi_list[@]}
  do
    case "${oneapi}" in
        compiler)
          installed_status_oneapi "libsycl.so" "Intel(R) OneAPI DPC++/C++ Compiler" "CMPLR_ROOT"
          ;;
        mkl)
          installed_status_oneapi "libmkl_sycl.so" "Intel(R) OneAPI Math Kernel Library" "MKLROOT"
          ;;
        tbb)
          installed_status_oneapi "libtbb.so" "Intel(R) OneAPI Threading Building Blocks" "TBBROOT"
          ;;
        ccl)
          if [[ ${IS_DETAIL} -eq 1 ]]; then
            installed_status_oneapi "libccl.so" "Intel(R) OneAPI Collective Communications Library" "CCL_ROOT"
          fi
          ;;
        esac
  done

  if [[ ${IS_FAILED} -eq 1 ]]; then
    echo ""
    echo -e "================= \033[31m Check Intel OneApi Failed \033[0m =================="
    echo ""
  else
    echo ""
    echo -e "================= \033[32m Check Intel OneApi Passed \033[0m =================="
    echo ""
  fi
}

check_python() {
  echo ""
  echo -e "======================== \033[33m Check Python \033[0m ========================"
  echo ""

  python_bin_path=$(which python || which python3 2>/dev/null)
  if [[ ${python_bin_path} = "" ]]; then
    die "Python is not installed" "Check Python Failed"
  fi

  v1=$(python --version|awk -F '[ .]' '{print $2}')
  v2=$(python --version|awk -F '[ .]' '{print $3}')
  echo -e "\033[33m python$v1.$v2 is installed. \033[0m"

  if [[ $v1 -le 2 ]]; then
    echo -e "\033[31m Python2 is not supported, please install python3 ! \033[0m"
    echo -e "==================== \033[31m Check Python Failed \033[0m ====================="
    echo ""
  elif [[ $v2 -lt ${min_python_version} ]]; then
    echo -e "\033[31m Your python version is too low, please upgrade to 3.${min_python_version} or higher! \033[0m"
    echo -e "==================== \033[31m Check Python Failed \033[0m ====================="
    echo ""
  elif [[ $v2 -gt ${max_python_version} ]]; then
    echo -e "\033[31m Your python version is too high, please downgrade to 3.${max_python_version} or lower! \033[0m"
    echo -e "==================== \033[31m Check Python Failed \033[0m ====================="
    echo ""
  else
    echo -e "==================== \033[32m Check Python Passed \033[0m ====================="
    echo ""
  fi
}

check_tensorflow() {
  echo ""
  echo -e "====================== \033[33m Check Tensorflow \033[0m ======================"
  echo ""

  v1=$(pip show tensorflow 2>/dev/null|grep Version|awk -F '[ :.]' '{print $3}')
  v2=$(pip show tensorflow 2>/dev/null|grep Version|awk -F '[ :.]' '{print $4}')
  if [[ ${v2} = "" ]]; then
    echo -e "\033[31m tensorflow is not installed! \033[0m"
    echo -e "================== \033[31m Check Tensorflow Failed \033[0m ==================="
    echo ""
  elif [[ ${v2} -ge ${min_tensorflow_version} ]]; then
    echo -e "\033[33m tensorflow${v1}.${v2} is installed. \033[0m"
    echo ""
    echo -e "================== \033[32m Check Tensorflow Passed \033[0m ==================="
    echo ""
  else
    die "tensorflow${v1}.${v2} is not supported" "Check Tensorflow Failed"
  fi

}

check_python_lib() {
  echo ""
  echo -e "====================== \033[33m Check Python Libraries \033[0m ======================"
  echo ""
  echo -e "\033[33m Check Tensorflow requires: \033[0m"
  echo ""
  for lib in ${tf_require_list[@]}
  do
    res=$(pip list|grep -E ${lib})
    if [[ ! -z ${res} ]]; then
      echo -e "\033[33m $res . \033[0m"
    else
      echo -e "\033[31m $lib should be installed. \033[0m"
    fi
    echo ""
  done

  echo -e "\033[33m Check Intel(R) Extension for TensorFlow* requires: \033[0m"
  echo ""
  for lib in ${itex_require_list[@]}
  do
    res=$(pip list|grep -w -E ${lib})
    if [[ ! -z ${res} ]]; then
      echo -e "\033[33m $res . \033[0m"
    else
      echo -e "\033[31m $lib should be installed. \033[0m"
    fi
  done

  echo -e "================== \033[32m Check Python Libraries Passed \033[0m ==================="
  echo ""
}

#===================================================================================================#
#
#                   CHECK ENVIRONMENT FOR INTEL EXTENSION FOR TENSORFLOW
#
#===================================================================================================#
main() {
  if [ $IS_HELP -eq 1 ]; then
    usage
    exit 0
  fi

  cat << EOM

    Check Environment for Intel(R) Extension for TensorFlow*...

EOM

  check_os
  check_intel_gpu_driver
  check_python
  check_tensorflow
  check_intel_oneapi
  if [ ${IS_DETAIL} -eq 1 ]; then
    check_python_lib
  fi

}

main
