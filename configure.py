# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""configure script to get build parameters from user."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import os
import platform
import subprocess
import sys

# pylint: disable=g-import-not-at-top
try:
  from shutil import which
except ImportError:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


_DEFAULT_DPCPP_TOOLKIT_PATH = '/opt/intel/oneapi/compiler/latest/linux'
_DEFAULT_MKL_PATH='/opt/intel/oneapi/mkl/latest'
_DEFAULT_AOT_CONFIG = ''
_DEFAULT_GCC_TOOLCHAIN_PATH = ''
_DEFAULT_GCC_TOOLCHAIN_TARGET = ''
_DEFAULT_OCL_SDK_ROOT = ''

_DEFAULT_PROMPT_ASK_ATTEMPTS = 10

_ITEX_BAZELRC_FILENAME = '.itex_configure.bazelrc'
_ITEX_WORKSPACE_ROOT = ''
_ITEX_BAZELRC = ''
_ITEX_CURRENT_BAZEL_VERSION = None

_DENY_PATH_LIST = ['..', ';', '|', '$', "'", '%', '*', '&', ':', '?', '<', '>', 'http', 'ftp'] # pylint: disable=line-too-long

def path_filter(path):
  for p in _DENY_PATH_LIST:
    if p in path:
      return False
  return True

class UserInputError(Exception):
  pass


def is_linux():
  return platform.system() == 'Linux'


def remove_configure_file():
  if os.path.exists(_ITEX_BAZELRC_FILENAME):
    os.remove(_ITEX_BAZELRC_FILENAME)


def get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def symlink_force(target, link_name):
  """Force symlink, equivalent of 'ln -sf'.

  Args:
    target: items to link to.
    link_name: name of the link.
  """
  try:
    os.symlink(target, link_name)
  except OSError as e:
    if e.errno == errno.EEXIST:
      os.remove(link_name)
      os.symlink(target, link_name)
    else:
      raise e


def sed_in_place(filename, old, new):
  """Replace old string with new string in file.

  Args:
    filename: string for filename.
    old: string to replace.
    new: new string to replace to.
  """
  try:
    with open(filename, 'r') as f:
      filedata = f.read()
  finally:
    f.close()
  newdata = filedata.replace(old, new)
  try:
    with open(filename, 'w') as f:
      f.write(newdata)
  finally:
    f.close()


def write_to_bazelrc(line):
  try:
    with open(_ITEX_BAZELRC, 'a') as f:
      f.write(line + '\n')
  finally:
    f.close()


def write_action_env_to_bazelrc(var_name, var):
  write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))


def run_shell(cmd, allow_non_zero=False):
  """Running shell command with check."""
  def _checked_cmd(cmd):
    deny_list = [';', '&', '|', '`', '\r', '\n', '(', ')', '<', '>']
    if cmd is None:
      print('Empty command!')
      return None
    for c in deny_list:
      if c in cmd:
        print('Invalid command!')
        return None
    return str(cmd).strip()

  safe_cmd = _checked_cmd(cmd)
  if safe_cmd is None:
    remove_configure_file()
    sys.exit(-1)
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd)
  return output.decode('UTF-8').strip()

def check_safe_python_bin_path(python_bin_path):
  """Check whether python binary path is safe"""
  normal_flag = True
  if python_bin_path is None:
    normal_flag = False
  else:
    for c in _DENY_PATH_LIST:
      if c in python_bin_path:
        normal_flag = False
  if not normal_flag:
    remove_configure_file()
    raise Exception("Invalid python binary path!")

  path_list = filter(path_filter, str(python_bin_path).strip().split(os.sep))
  result = os.sep.join(path_list)
  if result == python_bin_path.strip():
    return result

  raise Exception("Invalid python binary path!")


def check_safe_python_lib_path(python_lib_path):
  """Check whether python library path is safe"""
  normal_flag = True
  if python_lib_path is None:
    normal_flag = False
  else:
    for c in _DENY_PATH_LIST:
      if c in python_lib_path:
        normal_flag = False
  if not normal_flag:
    remove_configure_file()
    raise Exception("Invalid python library path!")

  path_list = filter(path_filter, str(python_lib_path).strip().split(os.sep))
  result = os.sep.join(path_list)
  if result == python_lib_path.strip():
    return result

  raise Exception("Invalid python library path!")


def get_python_path(environ_cp, python_bin_path):
  """Get the python site package paths."""
  python_paths = []
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')

  checked_python_bin_path = check_safe_python_bin_path(python_bin_path)
  try:
    library_paths = run_shell([
        checked_python_bin_path, '-c',
        'import site; print("\\n".join(site.getsitepackages()))'
    ]).split('\n')
    user_paths = run_shell([
        checked_python_bin_path, '-m',
        'site', '--user-site'
    ]).split('\n')
  except subprocess.CalledProcessError:
    library_paths = [
        run_shell([
            checked_python_bin_path, '-c',
            'from distutils.sysconfig import get_python_lib;'
            'print(get_python_lib())'
        ])
    ]

  all_paths = set(python_paths + library_paths + user_paths)

  paths = []
  for path in all_paths:
    if os.path.isdir(path):
      tf_path = path + os.path.sep + "tensorflow"
      if os.path.exists(tf_path):
        paths.append(path)
  if len(paths) == 0:
    raise Exception("Tensorflow package not found! Please install it first!")
  return paths


def get_python_major_version(python_bin_path):
  """Get the python major version."""
  checked_python_bin_path = check_safe_python_bin_path(python_bin_path)
  if checked_python_bin_path is not None:
    return None

  return run_shell(
      [checked_python_bin_path, '-c', 'import sys; print(sys.version[0])'])


def setup_python(environ_cp):
  """Setup python related env variables."""
  # Get PYTHON_BIN_PATH, default is the current running python.
  default_python_bin_path = sys.executable
  while True:
    python_bin_path = default_python_bin_path
    print('Python binary path: %s\n' % python_bin_path)

    # Check if the path is valid
    checked_python_bin_path = check_safe_python_bin_path(python_bin_path)
    if (os.path.isfile(checked_python_bin_path) and
        os.access(checked_python_bin_path, os.X_OK)):
      break
    if not os.path.exists(checked_python_bin_path):
      print('Invalid python path: python binary cannot be found.')
    else:
      print('Provided python binary is not executable. Is it a python binary?')
    environ_cp['PYTHON_BIN_PATH'] = ''

  # Get PYTHON_LIB_PATH
  python_lib_path = environ_cp.get('PYTHON_LIB_PATH')
  if not python_lib_path:
    python_lib_paths = get_python_path(environ_cp, checked_python_bin_path)
    if environ_cp.get('USE_DEFAULT_PYTHON_LIB_PATH') == '1':
      python_lib_path = python_lib_paths[0]
      checked_python_lib_path = check_safe_python_lib_path(python_lib_path)
    else:
      print('Found possible Python library paths:')
      print(python_lib_paths)
      print('\n')
      default_python_lib_path = check_safe_python_lib_path(python_lib_paths[0])
      python_lib_path = default_python_lib_path
      checked_python_lib_path = check_safe_python_lib_path(python_lib_path)
    environ_cp['PYTHON_LIB_PATH'] = checked_python_lib_path

  _ = get_python_major_version(checked_python_bin_path)

  # Set-up env variables used by python_configure.bzl
  write_action_env_to_bazelrc('PYTHON_BIN_PATH', checked_python_bin_path)
  write_action_env_to_bazelrc('PYTHON_LIB_PATH', checked_python_lib_path)
  write_to_bazelrc('build --python_path=\"%s"' % checked_python_bin_path)
  environ_cp['PYTHON_BIN_PATH'] = checked_python_bin_path

  # If choosen python_lib_path is from a path specified in the PYTHONPATH
  # variable, need to tell bazel to include PYTHONPATH
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
    if python_lib_path in python_paths:
      write_action_env_to_bazelrc('PYTHONPATH', environ_cp.get('PYTHONPATH'))
  # check tensorflw >=2.10.0
  # not check tensorflow-estimator version
  package_list= subprocess.Popen(os.path.sep.join(checked_python_bin_path.split(os.path.sep)[:-1]) + os.path.sep + "pip" + " list | grep \"^tensorflow \"", shell=True, stdout=subprocess.PIPE).stdout.read().decode()
  tensorflow_list = package_list.splitlines()
  for line in tensorflow_list:
    if line.startswith("tensorflow  "):
        name, version = line.split()
        current_tensorflow_version = convert_version_to_int(version)
        min_tf_version = convert_version_to_int("2.10.0")
        if current_tensorflow_version < min_tf_version:
          print('Make sure you installed tensorflow version >= 2.10.0')
          sys.exit(1)
    else:
         print('Make sure you installed tensorflow version >= 2.10.0')
         sys.exit(1)
  # Write tools/python_bin_path.sh
  try:
    with open(
        os.path.join(_ITEX_WORKSPACE_ROOT,
                     'itex', 'tools', 'python_bin_path.sh'),
        'w') as f:
      f.write('export PYTHON_BIN_PATH="%s"' % checked_python_bin_path)
  finally:
    f.close()

def create_build_configuration(environ_cp):

  tf_header_dir = environ_cp['PYTHON_LIB_PATH'] + "/tensorflow/include"
  tf_shared_lib_dir = environ_cp['PYTHON_LIB_PATH'] + "/tensorflow/"

  write_action_env_to_bazelrc("TF_HEADER_DIR", tf_header_dir)
  write_action_env_to_bazelrc("TF_SHARED_LIBRARY_DIR", tf_shared_lib_dir)
  write_action_env_to_bazelrc("TF_CXX11_ABI_FLAG", 1)


def reset_configure_bazelrc():
  """Reset file that contains customized config settings."""
  try:
    with open(_ITEX_BAZELRC, 'w') as f:
      pass
  finally:
    f.close()


def cleanup_makefile():
  """Delete any leftover BUILD files from the Makefile build.

  These files could interfere with Bazel parsing.
  """
  makefile_download_dir = os.path.join(_ITEX_WORKSPACE_ROOT, 'tensorflow',
                                       'contrib', 'makefile', 'downloads')
  if os.path.isdir(makefile_download_dir):
    for root, _, filenames in os.walk(makefile_download_dir):
      for f in filenames:
        if f.endswith('BUILD'):
          os.remove(os.path.join(root, f))


def get_var(environ_cp,
            var_name,
            query_item,
            enabled_by_default,
            question=None,
            yes_reply=None,
            no_reply=None):
  """Get boolean input from user.

  If var_name is not set in env, ask user to enable query_item or not. If the
  response is empty, use the default.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.

  Returns:
    boolean value of the variable.

  Raises:
    UserInputError: if an environment variable is set, but it cannot be
      interpreted as a boolean indicator, assume that the user has made a
      scripting error, and will continue to provide invalid input.
      Raise the error to avoid infinitely looping.
  """
  if not question:
    question = ('Do you wish to build Intel® Extension for TensorFlow* '
                'with %s support?') % query_item
  if not yes_reply:
    yes_reply = ('%s support will be enabled for '
                 'Intel® Extension for TensorFlow*.') % query_item
  if not no_reply:
    no_reply = 'No %s' % yes_reply

  yes_reply += '\n'
  no_reply += '\n'

  if enabled_by_default:
    question += ' [Y/n]: '
  else:
    question += ' [y/N]: '

  var = environ_cp.get(var_name)
  if var is not None:
    var_content = var.strip().lower()
    true_strings = ('1', 't', 'true', 'y', 'yes')
    false_strings = ('0', 'f', 'false', 'n', 'no')
    if var_content in true_strings:
      var = True
    elif var_content in false_strings:
      var = False
    else:
      raise UserInputError(
          'Environment variable %s must be set as a boolean indicator.\n'
          'The following are accepted as TRUE : %s.\n'
          'The following are accepted as FALSE: %s.\n'
          'Current value is %s.' %
          (var_name, ', '.join(true_strings), ', '.join(false_strings), var))

  while var is None:
    user_input_origin = get_input(question)
    user_input = user_input_origin.strip().lower()
    if user_input == 'y':
      print(yes_reply)
      var = True
    elif user_input == 'n':
      print(no_reply)
      var = False
    elif not user_input:
      if enabled_by_default:
        print(yes_reply)
        var = True
      else:
        print(no_reply)
        var = False
    else:
      print('Invalid selection! Please input Y(y) or N(n).')
  return var


def set_build_var(environ_cp,
                  var_name,
                  query_item,
                  option_name,
                  enabled_by_default,
                  bazel_config_name=None):
  """Set if query_item will be enabled for the build.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set subprocess environment variable and write to .bazelrc if enabled.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    option_name: string for option to define in .bazelrc.
    enabled_by_default: boolean for default behavior.
    bazel_config_name: Name for Bazel --config argument to enable build feature.
  """

  var = str(int(get_var(environ_cp, var_name, query_item, enabled_by_default)))
  environ_cp[var_name] = var
  if var == '1':
    write_to_bazelrc('build:%s --define %s=true' %
                     (bazel_config_name, option_name))
    write_to_bazelrc('build --config=%s' % bazel_config_name)
  elif bazel_config_name is not None:
    # TODO(mikecase): Migrate all users of configure.py to use --config Bazel
    # options and not to set build configs through environment variables.
    write_to_bazelrc('build:%s --define %s=true' %
                     (bazel_config_name, option_name))


def set_action_env_var(environ_cp,
                       var_name,
                       query_item,
                       enabled_by_default,
                       question=None,
                       yes_reply=None,
                       no_reply=None):
  """Set boolean action_env variable.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set environment variable and write to .bazelrc.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.
  """
  var = int(
      get_var(environ_cp, var_name, query_item, enabled_by_default, question,
              yes_reply, no_reply))

  write_action_env_to_bazelrc(var_name, var)
  environ_cp[var_name] = str(var)


def convert_version_to_int(version):
  """Convert a version number to a integer that can be used to compare.

  Version strings of the form X.YZ and X.Y.Z-xxxxx are supported. The
  'xxxxx' part, for instance 'homebrew' on OS/X, is ignored.

  Args:
    version: a version to be converted

  Returns:
    An integer if converted successfully, otherwise return None.
  """
  version = version.split('-')[0]
  version_segments = version.split('.')
  # Treat "0.24" as "0.24.0"
  if len(version_segments) == 2:
    version_segments.append('0')
  for seg in version_segments:
    if not seg.isdigit():
      return None

  version_str = ''.join(['%03d' % int(seg) for seg in version_segments])
  return int(version_str)


def check_bazel_version(min_version):
  """Check installed bazel version is higher than min_version.

  Args:
    min_version: string for minimum bazel version (must exist!).

  Returns:
    The bazel version detected.
  """
  if which('bazel') is None:
    print('Cannot find bazel. Please install bazel.')
    sys.exit(0)
  curr_version = run_shell(
      ['bazel', '--batch', '--bazelrc=/dev/null', 'version'])

  for line in curr_version.split('\n'):
    if 'Build label: ' in line:
      curr_version = line.split('Build label: ')[1]
      break

  min_version_int = convert_version_to_int(min_version)
  curr_version_int = convert_version_to_int(curr_version)

  # Check if current bazel version can be detected properly.
  if not curr_version_int:
    print('WARNING: current bazel installation is not a release version.')
    print('Make sure you are running at least bazel %s' % min_version)
    return curr_version

  print('You have bazel %s installed.' % curr_version)

  if curr_version_int < min_version_int:
    print('Please upgrade your bazel installation to version %s or higher to '
          'build Intel® Extension for TensorFlow*!' % min_version)
    sys.exit(1)
  return curr_version


def set_cc_opt_flags():
  """Set up architecture-dependent optimization flags.

  Also append CC optimization flags to bazel.rc..

  Args:
    environ_cp: copy of the os.environ.
  """
  default_cc_opt_flags = '-march=native -Wno-sign-compare'
  for opt in default_cc_opt_flags.split():
    write_to_bazelrc('build:opt --copt=%s' % opt)
  # It should be safe on the same build host.
  write_to_bazelrc('build:opt --host_copt=-march=native')
  write_to_bazelrc('build:opt --define with_default_optimizations=true')

def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var,
                                    var_default, default_only=False):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  if default_only:
    return var_default
  var = environ_cp.get(var_name)
  if var is None:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def prompt_loop_or_load_from_env(environ_cp,
                                 var_name,
                                 var_default,
                                 ask_for_var,
                                 check_success,
                                 error_msg,
                                 suppress_default_error=False,
                                 n_ask_attempts=_DEFAULT_PROMPT_ASK_ATTEMPTS):
  """Loop over user prompts for an ENV param until receiving a valid response.

  For the env param var_name, read from the environment or verify user input
  until receiving valid input. When done, set var_name in the environ_cp to its
  new value.

  Args:
    environ_cp: (Dict) copy of the os.environ.
    var_name: (String) string for name of environment variable, e.g. "TF_MYVAR".
    var_default: (String) default value string.
    ask_for_var: (String) string for how to ask for user input.
    check_success: (Function) function that takes one argument and returns a
      boolean. Should return True if the value provided is considered valid. May
      contain a complex error message if error_msg does not provide enough
      information. In that case, set suppress_default_error to True.
    error_msg: (String) invalid response upon check_success(input) failure.
    suppress_default_error: (Bool) Suppress the above error message in favor of
      one from the check_success function.
    n_ask_attempts: (Integer) Number of times to query for valid input before
      raising an error and quitting.

  Returns:
    [String] The value of var_name after querying for input.

  Raises:
    UserInputError: if a query has been attempted n_ask_attempts times without
      success, assume that the user has made a scripting error, and will
      continue to provide invalid input. Raise the error to avoid infinitely
      looping.
  """
  default = environ_cp.get(var_name) or var_default
  full_query = '%s [Default is %s]: ' % (
      ask_for_var,
      default,
  )

  for _ in range(n_ask_attempts):
    val = get_from_env_or_user_or_default(environ_cp, var_name, full_query,
                                          default)
    if check_success(val):
      break
    if not suppress_default_error:
      print(error_msg)
    environ_cp[var_name] = None
  else:
    raise UserInputError('Invalid %s setting was provided %d times in a row. '
                         'Assuming to be a scripting mistake.' %
                         (var_name, n_ask_attempts))

  environ_cp[var_name] = val
  return val


def reformat_version_sequence(version_str, sequence_count):
  """Reformat the version string to have the given number of sequences.

  For example:
  Given (7, 2) -> 7.0
        (7.0.1, 2) -> 7.0
        (5, 1) -> 5
        (5.0.3.2, 1) -> 5

  Args:
      version_str: String, the version string.
      sequence_count: int, an integer.

  Returns:
      string, reformatted version string.
  """
  v = version_str.split('.')
  if len(v) < sequence_count:
    v = v + (['0'] * (sequence_count - len(v)))

  return '.'.join(v[:sequence_count])


def set_dpcpp_toolkit_path(environ_cp):
  """Set DPCPP_TOOLKIT_PATH."""

  def toolkit_exists(toolkit_path):
    """Check if a dpc++ toolkit path is valid."""
    sycl_rt_lib_path = 'lib/libsycl.so'

    sycl_rt_lib_path_full = os.path.join(toolkit_path, sycl_rt_lib_path)
    exists = os.path.exists(sycl_rt_lib_path_full)
    if not exists:
      print('Invalid DPC++ library path. %s cannot be found' %
            (sycl_rt_lib_path_full))
    return exists

  dpcpp_toolkit_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='DPCPP_TOOLKIT_PATH',
      var_default=_DEFAULT_DPCPP_TOOLKIT_PATH,
      ask_for_var=(
          'Please specify the location where DPC++ is installed.'),
      check_success=toolkit_exists,
      error_msg='Invalid DPC++ compiler path. libsycl.so cannot be found.',
      suppress_default_error=True)

  write_action_env_to_bazelrc('DPCPP_TOOLKIT_PATH',
                              dpcpp_toolkit_path)
  lib_path = '%s/lib:%s/compiler/lib/intel64_lin' %(
      dpcpp_toolkit_path,
      dpcpp_toolkit_path,
  )

  ld_lib_path = lib_path
  ld_library_path = os.getenv('LD_LIBRARY_PATH')
  if ld_library_path is not None and len(ld_library_path) > 0:
    ld_lib_path += ':' + ld_library_path

  library_path = os.getenv('LIBRARY_PATH')
  if library_path is not None and len(library_path) > 0:
    lib_path += ':' + library_path

  mkl_path = os.getenv('ONEAPI_MKL_PATH')
  if mkl_path is not None and len(mkl_path) > 0:
    mkl_lib = '%s/lib/intel64' % (mkl_path)
    lib_path += ':' + mkl_lib
  write_action_env_to_bazelrc('LD_LIBRARY_PATH',
                              ld_lib_path)
  write_action_env_to_bazelrc('LIBRARY_PATH',
                              lib_path)

def set_mkl_path(environ_cp):
  """Set MKL Path."""
  def valid_mkl_path(mkl_home):
    exists = (
        os.path.exists(os.path.join(mkl_home, 'include')) and
        (os.path.exists(os.path.join(mkl_home, 'lib'))))
    if not exists:
      print(
          'Invalid path to the MKL Toolkit. %s or %s cannot be found'
          % (os.path.join(mkl_home, 'include'),
             os.path.exists(os.path.join(mkl_home, 'lib'))))
    return exists
  mkl_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ONEAPI_MKL_PATH',
      var_default=_DEFAULT_MKL_PATH,
      ask_for_var='Please specify the MKL toolkit folder.',
      check_success=valid_mkl_path,
      error_msg='Invalid path to the MKL Toolkit.',
      suppress_default_error=True)
  write_action_env_to_bazelrc('ONEAPI_MKL_PATH', mkl_path)

def set_aot_config(environ_cp):
  """Set AOT_CONFIG."""

  def aot_exists(aot_configs):
    """Determinate whether aot config is valid"""

    if len(aot_configs) < 1:
      return True

    # check for security purpose only
    targets = aot_configs.split(",")
    for target in targets:
      if len(target) > 20:
        print('Invalid AOT target: {}'.format(target))
        return False
    return True

  aot_config = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='AOT_CONFIG',
      var_default=_DEFAULT_AOT_CONFIG,
      ask_for_var=(
        'Please specify the Ahead of Time(AOT) compilation platforms,'
        ' separate with "," for multi-targets.'),
      check_success=aot_exists,
      error_msg='Invalid AOT target!',
      suppress_default_error=True)

  write_action_env_to_bazelrc('AOT_CONFIG', aot_config)

def system_specific_test_config(env):
  """Add default build and test flags required for TF tests to bazelrc."""
  write_to_bazelrc('test --flaky_test_attempts=3')
  write_to_bazelrc('test --test_size_filters=small,medium')
  write_to_bazelrc(
      'test --test_tag_filters=-benchmark-test,-no_oss,-oss_serial')
  write_to_bazelrc('test --build_tag_filters=-benchmark-test,-no_oss')
  if env.get('TF_NEED_DPCPP', None) == '1':
    write_to_bazelrc('test --test_tag_filters=-no_gpu')
    write_to_bazelrc('test --build_tag_filters=-no_gpu')
    write_to_bazelrc('test --test_env=LD_LIBRARY_PATH')
  else:
    write_to_bazelrc('test --test_tag_filters=-gpu')
    write_to_bazelrc('test --build_tag_filters=-gpu')


def set_system_libs_flag(environ_cp):
  """Set system libraries flag into bazelrc file."""
  syslibs = environ_cp.get('TF_SYSTEM_LIBS', '')
  if syslibs:
    if ',' in syslibs:
      syslibs = ','.join(sorted(syslibs.split(',')))
    else:
      syslibs = ','.join(sorted(syslibs.split()))
    write_action_env_to_bazelrc('TF_SYSTEM_LIBS', syslibs)

  if 'PREFIX' in environ_cp:
    write_to_bazelrc('build --define=PREFIX=%s' % environ_cp['PREFIX'])
  if 'LIBDIR' in environ_cp:
    write_to_bazelrc('build --define=LIBDIR=%s' % environ_cp['LIBDIR'])
  if 'INCLUDEDIR' in environ_cp:
    write_to_bazelrc('build --define=INCLUDEDIR=%s' % environ_cp['INCLUDEDIR'])


def config_info_line(name, help_text):
  """Helper function to print formatted help text for Bazel config options."""
  print('\t--config=%-12s\t# %s' % (name, help_text))


def check_safe_workspace_path(workspace):
  """Check whether if the workspace path is safe"""
  normal_flag = True
  if workspace is None:
    normal_flag = False
  for c in _DENY_PATH_LIST:
    if c in workspace:
      normal_flag = False
  if not normal_flag:
    remove_configure_file()
    raise Exception("Invalid workspace path!")

  path_list = filter(path_filter, str(workspace).strip().split(os.sep))
  result = os.sep.join(path_list)
  if result == workspace.strip():
    return result

  raise Exception("Invalid workspace path!")


def main():
  global _ITEX_WORKSPACE_ROOT
  global _ITEX_BAZELRC
  global _ITEX_CURRENT_BAZEL_VERSION

  if not is_linux():
    print('Only support linux currently.')
    sys.exit(1)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default=os.path.abspath(os.path.dirname(__file__)),
      help='The absolute path to your active Bazel workspace.')

  _ITEX_WORKSPACE_ROOT = check_safe_workspace_path(
      os.path.abspath(os.path.dirname(__file__)))
  _ITEX_BAZELRC = os.path.join(_ITEX_WORKSPACE_ROOT, _ITEX_BAZELRC_FILENAME)

  # Make a copy of os.environ to be clear when functions and getting and setting
  # environment variables.
  environ_cp = dict(os.environ)

  current_bazel_version = check_bazel_version('5.3.0')
  _ITEX_CURRENT_BAZEL_VERSION = convert_version_to_int(current_bazel_version)

  reset_configure_bazelrc()

  cleanup_makefile()
  setup_python(environ_cp)
  create_build_configuration(environ_cp)

  set_action_env_var(environ_cp, 'TF_NEED_DPCPP', 'GPU', True)
  if environ_cp.get('TF_NEED_DPCPP') == '1':
    set_dpcpp_toolkit_path(environ_cp)
    set_aot_config(environ_cp)
    set_action_env_var(environ_cp, 'TF_NEED_MKL', 'MKL', False)
    if environ_cp.get('TF_NEED_MKL') == '1':
      set_mkl_path(environ_cp)
  else:
    print('Only CPU support is available for '
          'Intel® Extension for TensorFlow*.')

  set_cc_opt_flags()
  set_system_libs_flag(environ_cp)

  # Add a config option to build TensorFlow 2.0 API.
  write_to_bazelrc('build:v2 --define=tf_api_version=2')

  system_specific_test_config(os.environ)

  print('Preconfigured Bazel build configs. You can use any of the below by '
        'adding "--config=<>" to your build command. See .bazelrc for more '
        'details.')
  if environ_cp.get('TF_NEED_DPCPP') == '1':
    config_info_line('gpu', ('Build Intel® Extension for TensorFlow* '
                     'with GPU support.'))
  else:
    config_info_line('cpu', 'Build Intel® Extension for TensorFlow* '
                     'with CPU support.')


if __name__ == '__main__':
  main()
