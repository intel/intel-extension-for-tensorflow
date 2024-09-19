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
host_compiler_install_dir="%{host_compiler_install_dir}"

def system(cmd):
  """Invokes cmd with os.system()"""
  
  ret = os.system(cmd)
  if os.WIFEXITED(ret):
    return os.WEXITSTATUS(ret)
  else:
    return -os.WTERMSIG(ret)

def call_compiler(argv, link = False, dpcpp = True, xetla = False, cpu_only = False, gpu_only = False):
  
  if xetla and not link:
    dpcpp = True

  flags = argv

  if cpu_only:
    link = False
    dpcpp = False
    gpu_only = False

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
  common_flags = ['-fPIC', '-fexceptions']
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
  
  if xetla:
    compile_flags.append("-std=c++20")
    compile_flags.append("-DXETPP_NEW_XMAIN")
  else:
    compile_flags.append("-std=c++17")    

  # link flags
  link_flags = ['-fPIC']
  link_flags.append("-Wl,-no-as-needed")
  link_flags.append("-Wl,--enable-new-dtags")
  link_flags.append("-Wl,-rpath=%{TF_SHARED_LIBRARY_DIR}/python")
  link_flags.append("-L%{TF_SHARED_LIBRARY_DIR}/python/")
  if link and gpu_only and len(AOT_DEVICE) > 0 and not xetla:
    link_flags.append("-fsycl-targets=spir64_gen,spir64")
    link_flags.append(AOT_DEVICE)
  if gpu_only:
    if xetla:
      link_flags.append('-Xs "-doubleGRF -Xfinalizer -printregusage  -Xfinalizer -DPASTokenReduction  -Xfinalizer -enableBCR"')
    else:
      link_flags.append('-Xs \'-options "-cl-poison-unsupported-fp64-kernels -cl-intel-enable-auto-large-GRF-mode"\'')
    link_flags.append('-lsycl')
    link_flags.append("-fsycl")
    # TODO use bazel --jobs number here.
    link_flags.append('-fsycl-max-parallel-link-jobs=8')
    link_flags.append("-Wl,-rpath=%{DPCPP_ROOT_DIR}/lib/")
    link_flags.append("-Wl,-rpath=%{DPCPP_ROOT_DIR}/compiler/lib/intel64_lin/")
    link_flags.append("-lze_loader")
    link_flags.append("-lOpenCL")

  # oneMKL config
  if '%{ONEAPI_MKL_PATH}' and gpu_only:
    common_flags.append('-DMKL_ILP64')
    common_flags.append('-isystem %{ONEAPI_MKL_PATH}/include')
    link_flags.append("-L%{ONEAPI_MKL_PATH}/lib/intel64")
    link_flags.append("-lmkl_sycl_blas")
    link_flags.append("-lmkl_sycl_lapack")
    link_flags.append("-lmkl_sycl_dft")
    link_flags.append("-lmkl_intel_ilp64")
    link_flags.append("-lmkl_sequential")
    link_flags.append("-lmkl_core")

  # link standard libraries(such as libstdc++) from configured python enviroment
  std_lib_path = '%{PYTHON_LIB_PATH}' +  '{0[0]}..{0[1]}..{0[2]}'.format([os.path.sep] * 3)
  link_flags.append("-L" + std_lib_path)

  flags += common_flags
  if link:
    flags += link_flags
  if dpcpp:
    flags += compile_flags

  def is_valid_flag(f, pure_host=False, gpu_build=True):
    if not gpu_build:
      _INVALID_FLAG = ['linux_prod', r'%{dpcpp_compiler_root}/lib/clang', r'%{dpcpp_compiler_root}/compiler/include', r'%{dpcpp_compiler_root}/opt/compiler/include', 'ITEX_USE_MKL', 'EIGEN_USE_DPCPP_BUILD', 'EIGEN_USE_DPCPP', 'EIGEN_USE_GPU']
    elif not pure_host:
      # filter out 'linux_prod' 'lib/clang' for host can't use std include files of DPC++ compiler
      _INVALID_FLAG = ['linux_prod', 'fsycl', 'fhonor', r'.cpp', r'.cc', r'.hpp', r'.h', '-o', 'EIGEN_USE_DPCPP_BUILD', 'ffp', r'%{dpcpp_compiler_root}/lib/clang']
    else:
      _INVALID_FLAG = [r'%{dpcpp_compiler_root}/lib/clang', r'%{dpcpp_compiler_root}/compiler/include', r'%{dpcpp_compiler_root}/opt/compiler/include', 'EIGEN_USE_DPCPP_BUILD']
    flag = True
    for i in _INVALID_FLAG:
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

  sycl_host_compile_flags = [f for f in flags if is_valid_flag(f)]
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

  if dpcpp or link or gpu_only:
    # ref: https://github.com/intel/llvm/blob/sycl/clang/docs/UsersManual.rst#controlling-floating-point-behavior
    flags.append('-fno-finite-math-only')
    flags.append('-fno-approx-func')
    if gpu_only:
      flags.append('-DINTEL_GPU_ONLY')
    if os.path.basename(DPCPP_PATH) == "clang":
      AVX_FLAG = ' -mfma -mavx -mavx2 '
    else:
      AVX_FLAG = ' -axCORE-AVX2 '
    GCC_INSTALL_DIR = ' --gcc-install-dir=' + host_compiler_install_dir + ' '
    cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + DPCPP_PATH + GCC_INSTALL_DIR + AVX_FLAG + ' '.join(flags))
  else:
    if cpu_only:
      flags.append('-DINTEL_CPU_ONLY')
    flags = [f for f in flags if is_valid_flag(f, True, False)]
    for i, f in enumerate(flags):
      if (i < len(flags) - 1)  and (f == '-isystem' or f == '-iquote'):
        while(flags[i+1].startswith('-')):
          flags.pop(i)
    cmd = ('env ' + ' ' + HOST_COMPILER_PATH + ' ' + ' '.join(flags))
    if '-Wl,-no-as-needed' in flags:
      # '-no-as-needed' option affects ELF DT_NEEDED tags for dynamic libraries mentioned on the command line AFTER the '-no-as-needed' option.
      # Add '-no-as-needed' option at the beginning of command line, to make sure all the dynamic library mentioned on the command line to be added a DT_NEEDED tag by linker
      cmd = ('env ' + ' ' + HOST_COMPILER_PATH + ' -Wl,-no-as-needed -mfma -O3 -mavx -mavx2 ' + ' '.join(flags))
    else:
      cmd = ('env ' + ' ' + HOST_COMPILER_PATH + ' -mfma -O3 -mavx -mavx2 ' + ' '.join(flags))

  return system(cmd)

def main():
  parser = ArgumentParser()
  parser.add_argument('--xetla', action='store_true')
  parser.add_argument('-dpcpp_compile', action='store_true')
  parser.add_argument('-link_stage', action='store_true')  
  parser.add_argument('-DINTEL_CPU_ONLY', action='store_true')
  parser.add_argument('-DINTEL_GPU_ONLY', action='store_true')
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
    return call_compiler(leftover, link=True, dpcpp=args.dpcpp_compile, cpu_only=args.DINTEL_CPU_ONLY, xetla=args.xetla, gpu_only=True)
  else:
    # compile for DPC++ object
    return call_compiler(leftover, link=False, dpcpp=args.dpcpp_compile, cpu_only=args.DINTEL_CPU_ONLY, xetla=args.xetla, gpu_only=True)

if __name__ == '__main__':
  sys.exit(main())
