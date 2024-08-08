
def getConfig(url, filename):
  import json
  import os
  import wget
  if os.path.exists(filename) is False:
    wget.download(url, filename)
  with open(filename, 'r') as f:
    config = json.load(f)
    return config

def check_python(config):
  print("Check Python")
  import importlib.util
  import sys

  itex_found = importlib.util.find_spec("intel_extension_for_tensorflow")
  if itex_found is None:
    exit("\033[31mPlease Install Intel(R) Extension for TensorFlow* first\033[0m.\n")
  
  location = ''.join(itex_found.submodule_search_locations) + "/python/"
  sys.path.append(location)
  try:
    from version import __version__
    if __version__ > config['latest_release']:
      itex_version = "latest"
    else:
      itex_version = __version__
  except Exception:
    print("Intel(R) Extension for TensorFlow* Version is Unknown.\n")
  python_major_version = sys.version_info.major
  python_minor_version = sys.version_info.minor
  python_micro_version = sys.version_info.micro
  if python_major_version < 2 :
    exit("\033[31mPython2 is not supported, please install Python3!\033[0m")
  elif python_minor_version < config['python_version']['min_python_version'][itex_version]:
    exit("\033[31mYour Python version is too low, please upgrade to 3.\033[0m" + 
    str(config['python_version']['min_python_version'][itex_version]) + "\033[31m or higher!\033[0m")
  elif python_minor_version > config['python_version']['max_python_version'][itex_version]:
    exit("\033[31mYour Python version is too high, please downgrade to 3.\033[0m" + 
    str(config['python_version']['max_python_version'][itex_version]) + "\033[31m or lower!\033[0m")
  print("\t Python " + str(python_major_version) + "." + str(python_minor_version) + "." + 
        str(python_micro_version) + " is Supported.")
  print("\033[32mCheck Python Passed\033[0m\n")
  return itex_version

def check_os(config, itex_version):
  print("Check OS")
  import platform
  system_type = platform.system()
  if system_type != 'Linux':
    exit("\033[31mWe only Support Linux System!\033[0m\n")

  with open('/etc/os-release', 'r') as f:
    for line in f:
      if line.startswith('NAME='):
        os_id = line.strip().split('=')[1].lower().strip('"')
      if line.startswith('VERSION_ID='):
        os_version = line.strip().split('=')[1].strip('"')
  if os_id in config['os_list']:
    if os_version in config['os_version'][os_id][itex_version]:
      print("\tOS " + os_id + ":" + os_version + " is Supported")
  else:
    exit("\033[31mIntel GPU Driver Does Not Support OS \033[0m" + os_id + " : " + os_version + "\n \033[33mCheck OS failed. \033[0m\n")
  print("\033[32mCheck OS Passed\033[0m\n")
  return os_id, os_version

def check_tensorflow(config, itex_version):
  print("Check Tensorflow")
  import subprocess
  import importlib
  import os

  tf_found = importlib.util.find_spec("tensorflow")
  if tf_found is None:
    exit("\033[31mPlease Install TensorFlow first.\033[0m\n")
  
  #file = ''.join(tf_found.submodule_search_locations) + "/tools/pip_package/setup.py"
  #cmd = "cat " + file + '|grep "_VERSION ="'
  #res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  #tf_version = str(res.stdout)[14:20]
  import tensorflow as tf
  tf_version = tf.__version__
  print("\tTensorflow " + tf_version + " is installed.")
  if tf_version < config['tensorflow_version']['min_tensorflow_version'][itex_version]:
    exit("Your Tensorflow version is too low, please upgrade to " + 
         str(config['tensorflow_version']['min_tensorflow_version'][itex_version]) + "!")
  print("Check Tensorflow Passed\n")

def check_driver(config, os_id):
  print("Check Intel GPU Driver")
  import os
  # tf_version = str(res.stdout)[14:20]
  if os_id == "ubuntu":
    cmd = "dpkg -s "
  elif os_id == "rhel":
    cmd = "yum info installed "
    print(res)
  elif os_id == "sles":
        cmd = "zypper se --installed-only "
  else:
    exit("\033[31mUnsupported OS \n Check Intel GPU Driver Failed\033[0m\n")

  for driver in config['intel_gpu_driver_list'][os_id]:
      if os.system(cmd + driver) != 0:
        exit("\033[31m\tCheck Intel GPU Driver Failed\033[0m\n")

  print("Check Intel GPU Driver Passsed\n")

def check_oneapi(config, itex_version):
  print("Check OneAPI")
  import subprocess
  import os
  cmd = 'LD_DEBUG=libs python -c "import intel_extension_for_tensorflow" 2>&1 |tee /tmp/log' 
  subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  for oneapi in config['oneapi']:
    for lib in config['oneapi_lib'][oneapi]:
      cmd = 'grep ' + lib + ' /tmp/log'
      res = os.system(cmd)
      if res != 0:
        exit("\033[31m\tCan't find \033[0m" + lib + "\n\033[31m Check OneAPI Failed\033[0m\n" )
    print("\t" + config['oneapi'][oneapi] + " is Installed.")
    print("Recommended " + oneapi + " version is " + config['oneapi_version'][itex_version][oneapi])
  print("\033[32mCheck OneAPI Passed\033[0m\n")

def check_py_lib(config):
  print("Check Tensorflow Requirements\n")
  import importlib
  for lib in config['tf_requirements']:
    lib_found = importlib.util.find_spec(lib)
    if lib_found is None:
      print("\t" + lib + "\033[31m should be installed.\033[0m")
  print("\033[32mCheck Intel(R) Extension for TensorFlow* Requirements Passed\033[0m")
  for lib in config['itex_requirements']:
    lib_found = importlib.util.find_spec(lib)
    if lib_found is None:
      print("\t" + lib + "\033[31m should be installed.\033[0m")
  print("\n")

def check():
  print("\nCheck Environment for Intel(R) Extension for TensorFlow*...\n")
  print('__file__:    ', __file__)
  import configparser
  import os

  # the latest version
  url="https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/tools/python/config.json"

  # the local version
  configfile="config.json"
  configfile_path=os.path.dirname(__file__) + os.sep + configfile
  os_id=""
  config = getConfig(url, configfile_path)
  itex_version = check_python(config)
  os_id, os_version = check_os(config, itex_version)
  check_tensorflow(config, itex_version)
  check_driver(config, os_id)
  check_oneapi(config, itex_version)
  check_py_lib(config)

if __name__ == '__main__':
  check()