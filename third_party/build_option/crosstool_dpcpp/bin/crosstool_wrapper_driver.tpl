#!/usr/bin/env python

"""Crosstool wrapper for compiling DPC++ program
SYNOPSIS:
  crosstool_wrapper_driver [options passed in by cc_library()
                            or cc_binary() rule]

DESCRIPTION:
  call DPC++ compiler for device-side code, and host
  compiler for other code.
"""

from __future__ import print_function
from argparse import ArgumentParser
import os
import subprocess
import re
import sys
import pipes

TMPDIR = "%{TMP_DIRECTORY}"

os.environ["TMPDIR"] = TMPDIR
os.environ["TEMP"] = TMPDIR
os.environ["TMP"] = TMPDIR

if not os.path.exists(TMPDIR):
  os.makedirs(TMPDIR, exist_ok=True)

def check_is_intel_llvm(path):
  cmd = path + " -dM -E -x c /dev/null | grep '__INTEL_LLVM_COMPILER'"
  check_result = subprocess.getoutput(cmd)
  if len(check_result) > 0 and check_result.find('__INTEL_LLVM_COMPILER') > -1:
    return True
  return False

DPCPP_PATH = os.path.join("%{dpcpp_compiler_root}", "bin/icx")

if not os.path.exists(DPCPP_PATH):
  DPCPP_PATH = os.path.join('%{dpcpp_compiler_root}', 'bin/clang')
  if not os.path.exists(DPCPP_PATH) or check_is_intel_llvm(DPCPP_PATH):
    raise RuntimeError("compiler not found or invalid")

HOST_COMPILER_PATH = "%{HOST_COMPILER_PATH}"
DPCPP_COMPILER_VERSION = "%{DPCPP_COMPILER_VERSION}"

def system(cmd):
  """Invokes cmd with os.system()"""
  
  ret = os.system(cmd)
  if os.WIFEXITED(ret):
    return os.WEXITSTATUS(ret)
  else:
    return -os.WTERMSIG(ret)

def call_compiler(argv, link = False, dpcpp = True):
  flags = argv

  # TODO(itex): check dose this compiler has
  # -fno-sycl-use-footer flag. Once we
  # totally move to new compiler, we should
  # remove this part of code.
  has_fno_sycl_use_footer = False
  check_cmd = DPCPP_PATH + ' --help | grep fno-sycl-use-footer'
  check_result = subprocess.getoutput(check_cmd)
  if len(check_result) > 0 and check_result.find('fno-sycl-use-footer') > -1:
    has_fno_sycl_use_footer = True

# common flags
  common_flags = ['-fPIC']
  sycl_device_only_flags = ['-fsycl']
  sycl_device_only_flags.append('-fno-sycl-unnamed-lambda')
  sycl_device_only_flags.append('-fsycl-targets=spir64_gen,spir64')
  # TODO(itex): disable for SUSE regression
  #sycl_device_only_flags.append('-fsycl-host-compiler=' + HOST_COMPILER_PATH)
  sycl_device_only_flags.append('-sycl-std=2020')
  sycl_device_only_flags.append('-fhonor-infinities')
  sycl_device_only_flags.append('-fhonor-nans')
  AOT_DEVICE = ["%{AOT_DEVICES}"]
  AOT_DEVICE = AOT_DEVICE if AOT_DEVICE[0] != "" else []

  if has_fno_sycl_use_footer and dpcpp:
    sycl_device_only_flags.append('-fno-sycl-use-footer')

  sycl_device_only_flags.append('-Xclang -fdenormal-fp-math=preserve-sign')
  sycl_device_only_flags.append('-Xclang -cl-mad-enable')
  sycl_device_only_flags.append('-cl-fp32-correctly-rounded-divide-sqrt')
  sycl_device_only_flags.append('-fsycl-device-code-split=per_source')
  compile_flags = []
  compile_flags.append(' -isystem ' + ' -isystem '.join(%{dpcpp_builtin_include_directories}))
  compile_flags.append('-DDNNL_GRAPH_WITH_SYCL=1')

# link flags
  link_flags = ['-fPIC']
  link_flags.append('-lsycl')
  link_flags.append("-fsycl")
  link_flags.append('-Xs \'-options "-cl-poison-unsupported-fp64-kernels -cl-intel-enable-auto-large-GRF-mode"\'')
  # TODO use bazel --jobs number here.
  link_flags.append('-fsycl-max-parallel-link-jobs=8')
  link_flags.append("-Wl,-no-as-needed")
  link_flags.append("-Wl,-rpath=%{DPCPP_ROOT_DIR}/lib/")
  link_flags.append("-Wl,-rpath=%{DPCPP_ROOT_DIR}/compiler/lib/intel64_lin/")
  link_flags.append("-Wl,-rpath=%{TF_SHARED_LIBRARY_DIR}/python")
  link_flags.append("-L%{TF_SHARED_LIBRARY_DIR}/python/")
  link_flags.append("-lze_loader")
  link_flags.append("-lOpenCL")
  # link standard libraries(such as libstdc++) from configured python enviroment
  std_lib_path = '%{PYTHON_LIB_PATH}' +  '{0[0]}..{0[1]}..{0[2]}'.format([os.path.sep] * 3)
  link_flags.append("-L" + std_lib_path)
  if link and len(AOT_DEVICE) > 0:
    link_flags.append("-fsycl-targets=spir64_gen,spir64")
    link_flags.append(AOT_DEVICE)

# oneMKL config
  if '%{ONEAPI_MKL_PATH}':
    compile_flags.append('-DMKL_ILP64')
    compile_flags.append('-isystem %{ONEAPI_MKL_PATH}/include')
    link_flags.append("-L%{ONEAPI_MKL_PATH}/lib/intel64")
    link_flags.append("-lmkl_sycl")
    link_flags.append("-lmkl_intel_ilp64")
    link_flags.append("-lmkl_tbb_thread")
    link_flags.append("-lmkl_core")

  flags += common_flags
  if link:
    flags += link_flags
  if dpcpp:
    flags += compile_flags

  def is_vaild_flag(f):
    # filter out 'linux_prod' 'lib/clang' for host can't use std include files of DPC++ compiler
    _INVAILD_FLAG = ['linux_prod', 'fsycl', 'fhonor', r'.cpp', r'.cc', r'.hpp', r'.h', '-o', r'.o', 'EIGEN_USE_DPCPP_BUILD', 'ffp', r'lib/clang']
    flag = True
    for i in _INVAILD_FLAG:
      if i[0] == '-':
        if f.strip() == i:
          flag = False
          break
      else:
        if i in f:
          flag = False
          break
    return flag

  for i, f in enumerate(flags):
    if isinstance(f, list):
      flags[i] = ''.join(f)

  sycl_host_compile_flags = [f for f in flags if is_vaild_flag(f)]
  sycl_host_compile_flags.append('-std=c++17')
  # TODO(itex): disable for SUSE regression
  #host_flags = '-fsycl-host-compiler-options=\"%s"' % (' '.join(sycl_host_compile_flags))

  if dpcpp:
    flags += sycl_device_only_flags
  # TODO(itex): disable for SUSE regression
  #if dpcpp:
  #  flags.append(host_flags)

  for i, f in enumerate(flags):
    if isinstance(f, list):
      flags[i] = ''.join(f)

  cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + DPCPP_PATH + ' ' + ' '.join(flags))

  return system(cmd)

def main():
  parser = ArgumentParser()
  parser.add_argument('-dpcpp_compile', action='store_true')
  parser.add_argument('-link_stage', action='store_true')
  if len(sys.argv[1:]) == 1 and (sys.argv[1:][0].startswith('@')):
    with open(sys.argv[1:][0].split('@')[1],'r') as file:
      real_args = file.readlines()
      real_args = [x.strip() for x in real_args]
      args, leftover = parser.parse_known_args(real_args)
  else:
    args, leftover = parser.parse_known_args(sys.argv[1:])

  leftover = [pipes.quote(s) for s in leftover]
  if args.link_stage:
    # link for DPC++ object
    return call_compiler(leftover, link=True, dpcpp=args.dpcpp_compile)
  else:
    # compile for DPC++ object
    return call_compiler(leftover, link=False, dpcpp=args.dpcpp_compile)

if __name__ == '__main__':
  sys.exit(main())
