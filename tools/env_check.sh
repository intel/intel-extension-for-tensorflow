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

declare -A ubuntu_version_list
ubuntu_version_list[1.0.0]="20.04"
ubuntu_version_list[1.1.0]="20.04 22.04"
ubuntu_version_list[1.2.0]="20.04 22.04"
ubuntu_version_list[2.13.0]="20.04 22.04"
ubuntu_version_list[latest]="20.04 22.04"

declare -A redhat_version_list
redhat_version_list[1.0.0]="8.5"
redhat_version_list[1.1.0]="8.6"
redhat_version_list[1.2.0]="8.6"
redhat_version_list[2.13.0]="8.7 8.8"
redhat_version_list[latest]="8.7 8.8"

declare -A sles_version_list
sles_version_list[1.1.0]="15.3 15.4"
sles_version_list[1.2.0]="15.3 15.4"
sles_version_list[2.13.0]="15.3 15.4"
sles_version_list[latest]="15.3 15.4"

declare -A min_python_version
min_python_version[1.0.0]=7
min_python_version[1.1.0]=7
min_python_version[1.2.0]=8
min_python_version[2.13.0]=8
min_python_version[latest]=8

declare -A max_python_version
max_python_version[1.0.0]=10
max_python_version[1.1.0]=10
max_python_version[1.2.0]=11
max_python_version[2.13.0]=11
max_python_version[latest]=11

declare -A min_tensorflow_version
min_tensorflow_version[1.0.0]=10
min_tensorflow_version[1.1.0]=10
min_tensorflow_version[1.2.0]=12
min_tensorflow_version[2.13.0]=13
min_tensorflow_version[latest]=13

driver_list_for_ubuntu=(
  "intel-level-zero-gpu"
  "intel-opencl-icd"
  "level-zero"
  "libigc1"
  "libigdfcl1"
  "libigdgmm12"
)

driver_list_for_rhel=(
  "intel-igc-core"
  "intel-igc-opencl"
  "intel-gmmlib"
  "intel-opencl"
  "level-zero"
  "level-zero-devel"
)

driver_list_for_sles=(
  "intel-level-zero-gpu"
  "intel-opencl"
  "level-zero"
  "libigc1"
  "libigdfcl1"
  "libigdgmm12"
)

oneapi_list=(compiler mkl ccl)

# ITEX v1.0.0 GPU Driver Version
itex_1_0_driver_version_ubuntu=(
  "1.3.23726.1+i419"
  "22.28.23726.1+i419"
  "1.8.1+i419"
  "1.0.11485+i419"
  "1.0.11485+i419"
  "22.1.7+i419"
)

itex_1_0_driver_version_rhel=(
  "1.0.11485-i419.el8"
  "1.0.11485-i419.el8"
  "22.1.7-i419.el8"
  "22.28.23726.1-i419.el8"
  "1.8.1-i755.el8"
  "1.8.1-i755.el8"
)

# ITEX v1.1.0 GPU Driver Version
itex_1_1_driver_version_ubuntu=(
  "1.3.24595.35+i538"
  "22.43.24595.35+i538"
  "1.8.8+i524"
  "1.0.12504.6+i537"
  "1.0.12504.6+i537"
  "22.3.1+i529"
)

itex_1_1_driver_version_rhel=(
  "1.0.12504.6-i537.el8"
  "1.0.12504.6-i537.el8"
  "22.3.1-i529.el8"
  "22.43.24595.35-i538.el8"
  "1.8.8-i524.el8"
  "1.8.8-i524.el8"
)

itex_1_1_driver_version_sles=(
  "1.3.24595.35-i538"
  "22.43.24595.35-i538"
  "1.8.8-i524"
  "1.0.12504.6-i537"
  "1.0.12504.6-i537"
  "22.3.1-i529"
)

# ITEX v1.2.0 GPU Driver Version
itex_1_2_driver_version_ubuntu=(
  "1.3.25593.18-601"
  "23.05.25593.18-601"
  "1.9.4+i589"
  "1.0.13230.8-600"
  "1.0.13230.8-600"
  "22.3.5-601"
)

itex_1_2_driver_version_rhel=(
  "1.0.13230.8-i600.el8"
  "1.0.13230.8-i600.el8"
  "22.3.5-i601.el8" 
  "23.05.25593.18-i601.el8"
  "1.9.4-i589.el8"
  "1.9.4-i589.el8"
)

itex_1_2_driver_version_sles=(
  "1.3.25593.18-i601"
  "23.05.25593.18-i601"
  "1.9.4-i589"
  "1.0.13230.8-i600"
  "1.0.13230.8-i600"
  "22.3.5-i601"
)

# ITEX v2.13.0 GPU Driver Version
itex_1_3_driver_version_ubuntu=(
  "1.3.26241.33-647~22.04",
  "23.17.26241.33-647~22.04",
  "1.11.0-647~22.04",
  "1.0.13822.8-647~22.04",
  "1.0.13822.8-647~22.04",
  "22.3.5-647~22.04"
)

itex_1_3_driver_version_rhel=(
  "1.0.13822.8-647.el8",
  "1.0.13822.8-647.el8",
  "22.3.5-i647.el8",
  "23.17.26241.33-647.el8",
  "1.11.0-647.el8",
  "1.11.0-647.el8"
)

itex_1_3_driver_version_sles=(
  "1.3.26241.33-647",
  "23.17.26241.33-647",
  "1.11.0-i647",
  "1.0.13822.8-647",
  "1.0.13822.8-647",
  "22.3.5-i647"
)

declare -A driver_version_ubuntu
driver_version_ubuntu[1.0.0]=${itex_1_0_driver_version_ubuntu[@]}
driver_version_ubuntu[1.1.0]=${itex_1_1_driver_version_ubuntu[@]}
driver_version_ubuntu[1.2.0]=${itex_1_2_driver_version_ubuntu[@]}
driver_version_ubuntu[2.13.0]=${itex_1_3_driver_version_ubuntu[@]}

declare -A driver_version_rhel
driver_version_rhel[1.0.0]=${itex_1_0_driver_version_rhel[@]}
driver_version_rhel[1.1.0]=${itex_1_1_driver_version_rhel[@]}
driver_version_rhel[1.2.0]=${itex_1_2_driver_version_rhel[@]}
driver_version_rhel[1.3.0]=${itex_1_3_driver_version_rhel[@]}

declare -A driver_version_sles
driver_version_sles[1.1.0]=${itex_1_1_driver_version_sles[@]}
driver_version_sles[1.2.0]=${itex_1_2_driver_version_sles[@]}
driver_version_sles[1.3.0]=${itex_1_3_driver_version_sles[@]}

itex_1_0_oneapi_version=(
  "2022.2.0-8734"
  "2022.2.0-8748"
)

itex_1_1_oneapi_version=(
  "2023.0.0-25370"
  "2023.0.0-25398"
  "2021.8.0-25371"
)

itex_1_2_oneapi_version=(
  "2023.1.0-46305"
  "2023.1.0-46342"
  "2021.9.0-43543"
)

itex_1_3_oneapi_version=(
  "2023.2.0-49495"
  "2023.2.0-49495"
  "2021.10.0-49084"
)

declare -A oneapi_version
oneapi_version[1.0.0]=${itex_1_0_oneapi_version[@]}
oneapi_version[1.1.0]=${itex_1_1_oneapi_version[@]}
oneapi_version[1.2.0]=${itex_1_2_oneapi_version[@]}
oneapi_version[2.13.0]=${itex_1_3_oneapi_version[@]}

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
  -d, --detail       print Tensorflow and Intel® Extension for TensorFlow* required python libraries' installed status.

EOM
}

info() {
  echo -e "\033[33m $1. \033[0m"
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
        if [[ "${ubuntu_version_list[$itex_version]}"  =~ "${os_version}" ]]; then
          info "OS ${os_name}:${os_version} is Supported"
        else
          die "Intel GPU driver does not support OS ${os_name}:${os_version} yet" " Check OS Failed"
        fi
        ;;
      rhel)
        if [[ "${redhat_version_list[$itex_version]}"  =~ "${os_version}" ]]; then
          info "OS ${os_name}:${os_version} is Supported"
        else
          die "Intel GPU driver does not support OS ${os_name}:${os_version} yet" " Check OS Failed"
        fi
        ;;
      sles)
        if [[ "${sles_version_list[$itex_version]}"  =~ "${os_version}" ]]; then
          info "OS ${os_name}:${os_version} is Supported"
        else
          die "Intel GPU driver does not support OS ${os_name}:${os_version} yet" " Check OS Failed"
        fi
        ;;
      *)
        die "Unknow OS ${os_name}" " Check OS Failed"
        ;;
  esac

  echo ""
  echo -e "====================== \033[32m Check OS Passed \033[0m ======================="
  echo ""
}

installed_status_driver() {
  case "${os_name}" in
    ubuntu)
      driver_list=(${driver_version_ubuntu[$itex_version]// / })
      status=$(dpkg -s $1 2>/dev/null |grep Status|awk -F ':' '{print $2}'|grep "install ok installed"|sed 's/^\s*//g')
      version=$(dpkg -s $1 2>/dev/null |grep Version|awk -F ':|~' '{print $2}'|sed 's/^\s*//g')
      ;;
    rhel)
      driver_list=(${driver_version_rhel[$itex_version]// / })
      status=$(yum info installed $1 2>/dev/null|grep Name|awk -F ':' '{print $2}')
      version=$(yum info installed $1 2>/dev/null|grep Version|awk -F ':' '{print $2}'|sed 's/^\s*//g')
      ;;
    sles)
      driver_list=(${driver_version_sles[$itex_version]// / })
      status=$(zypper se --installed-only |grep $1)
      version=$(rpm -qa --info $1 2>/dev/null|grep Version|awk -F ':' '{print $2}'|sed 's/^\s*//g')
      ;;
    *)
      echo -e "=============== \033[31m Check Intel GPU Driver Failed \033[0m ================"
      ;;
  esac
  if [[ -z $status ]]; then
    echo -e "\033[31m Intel(R) graphics runtime $1 is not installed! \033[0m"
  elif [[ ! "${driver_list[@]}" =~ "$version" ]]; then
    info "Intel(R) graphics runtime $1-${version} is installed, but is not recommended ${driver_list[$2]}"
  else
    info "Intel(R) graphics runtime $1-${version} is installed"
  fi

}

check_intel_gpu_driver() {
  echo ""
  echo -e "=================== \033[33m Check Intel GPU Driver \033[0m ==================="
  echo ""

  case "${os_name}" in
    ubuntu)
      driver_list=${driver_list_for_ubuntu[@]}
      ;;
    rhel)
      driver_list=${driver_list_for_rhel[@]}
      ;;
    sles)
      driver_list=${driver_list_for_sles[@]}
      ;;
    *)
      echo -e "=============== \033[31m Check Intel GPU Driver Failed \033[0m ================"
      ;;
  esac
  i=0
  for driver in ${driver_list[@]}
  do
    installed_status_driver ${driver} $((i++))
  done
  echo ""
  echo -e "=============== \033[32m Check Intel GPU Driver Finshed \033[0m ================"
  echo ""
}

check_device_availability() {
  echo ""
  echo -e "========================== \033[33m Check Devices Availability \033[0m =========================="
  echo ""
  device_list=$(python -c "import tensorflow as tf;print(tf.config.list_physical_devices())")
  if [[ ! ${device_list[@]} =~ "XPU" ]]; then
    os_glib_path=(`find /usr/lib /usr/local/lib /usr/lib64 -name libstdc++.so.6.*`)
    glib_path=(${os_glib_path[@]})
    if [[ ! -z ${CONDA_PREFIX} ]]; then
      conda_glib_path=`find ${CONDA_PREFIX} ${CONDA_PREFIX_1}/lib -name libstdc++.so.6.*`
      glib_path=(${os_glib_path[@]} ${conda_glib_path[*]})
    fi

    info "You have multiple libstdc++.so.6, make sure you are using the correct one"
    for path in ${glib_path[@]};
    do
      info "    $path"
    done
    echo ""

    die "Enable OCL_ICD_ENABLE_TRACE=1 OCL_ICD_DEBUG=2 to obtain detail information when using Intel® Extension for TensorFlow*" "Check Devices Availability Failed"
  fi
  echo ""
  echo -e "====================== \033[32m Check Devices Availability Passed \033[0m ======================="
  echo ""
}

installed_status_oneapi() {
  grep $1 ${LOAD_LIBS}|grep "fini" -q 2>&1
  if [ $? -eq 0 ]; then
    echo -e "\033[33m $2 is installed. \033[0m"
  else
    echo -e "\033[31m Can't find $1, $2 is uninstalled supported version $4 or unset relevant environment variables, such as $3. \033[0m"
    IS_FAILED=1
  fi
}

check_intel_oneapi() {
  echo ""
  echo -e "===================== \033[33m Check Intel oneAPI \033[0m ====================="
  echo ""

  LOAD_LIBS=/tmp/loadlibs
  LD_DEBUG=libs ${python_bin_path} -c "import intel_extension_for_tensorflow"  2>>${LOAD_LIBS}
  current_oneapi_list=(${oneapi_version[$itex_version]// / })
  for oneapi in ${oneapi_list[@]}
  do
    case "${oneapi}" in
        compiler)
          installed_status_oneapi "libsycl.so" "Intel(R) oneAPI DPC++/C++ Compiler" "CMPLR_ROOT" ${current_oneapi_list[0]}
          ;;
        mkl)
          installed_status_oneapi "libmkl_sycl.so" "Intel(R) oneAPI Math Kernel Library" "MKLROOT" ${current_oneapi_list[1]}
          ;;
        ccl)
          if [[ ${IS_DETAIL} -eq 1 ]]; then
            installed_status_oneapi "libccl.so" "Intel(R) oneAPI Collective Communications Library" "CCL_ROOT" ${current_oneapi_list[2]}
          fi
          ;;
        esac
  done

  if [[ ${IS_FAILED} -eq 1 ]]; then
    echo ""
    echo -e "================= \033[31m Check Intel oneAPI Failed \033[0m =================="
    echo ""
  else
    echo ""
    echo -e "================= \033[32m Check Intel oneAPI Passed \033[0m =================="
    echo ""

    check_device_availability
  fi
}

check_python() {
  echo ""
  echo -e "======================== \033[33m Check Python \033[0m ========================"
  echo ""

  python_bin_path=$(which python 2>/dev/null|| which python3 2>/dev/null || die "Python is not installed" "Check Python Failed")

  itex_version=$(pip show intel_extension_for_tensorflow|grep Version|awk '{print $2}')
  itex_lib_version=$(pip show intel_extension_for_tensorflow_lib|grep Version|awk '{print $2}')

  if [ -z ${itex_version} ]; then
    die "Please install Intel(R) Extension for TensorFlow* first." "Check Failed"
  fi

  v1=$(${python_bin_path} --version|awk -F '[ .]' '{print $2}')
  v2=$(${python_bin_path} --version|awk -F '[ .]' '{print $3}')
  echo -e "\033[33m python$v1.$v2 is installed. \033[0m"

  if [ -z ${min_python_version[$itex_version]} ]; then
    itex_version=latest
  fi

  if [[ $v1 -le 2 ]]; then
    die "Python2 is not supported, please install python3!" "Check Python Failed"
  elif [[ $v2 -lt ${min_python_version[$itex_version]} ]]; then
    die "Your python version is too low, please upgrade to 3.${min_python_version[$itex_version]} or higher!" "Check Python Failed"
  elif [[ $v2 -gt ${max_python_version[$itex_version]} ]]; then
    die "Your python version is too high, please downgrade to 3.${max_python_version[$itex_version]} or lower!" "Check Python Failed"
  fi

  echo ""
  echo -e "==================== \033[32m Check Python Passed \033[0m ====================="
  echo ""
}

check_tensorflow() {
  echo ""
  echo -e "====================== \033[33m Check Tensorflow \033[0m ======================"
  echo ""

  v1=$(pip show tensorflow 2>/dev/null|grep Version|awk -F '[ :.]' '{print $3}')
  v2=$(pip show tensorflow 2>/dev/null|grep Version|awk -F '[ :.]' '{print $4}')
  echo -e "\033[33m Tensorflow${v1}.${v2} is installed. \033[0m"

  if [[ ${v2} = "" ]]; then
    die "Tensorflow is not installed!" "Check Tensorflow Failed"
  elif [[ ${v2} -lt ${min_tensorflow_version[$itex_version]} ]]; then
    die "Your Tensorflow version is too low, please upgrade to ${min_tensorflow_version[$itex_version]}!" "Check Tensorflow Failed"
  fi

  echo ""
  echo -e "================== \033[32m Check Tensorflow Passed \033[0m ==================="
  echo ""
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

  echo ""
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

  check_python
  check_os
  check_tensorflow

  if [ ${itex_lib_version: -1} -eq 1 ]; then
    check_intel_gpu_driver
    check_intel_oneapi
  fi

  if [ ${IS_DETAIL} -eq 1 ]; then
    check_python_lib
  fi

}

main
