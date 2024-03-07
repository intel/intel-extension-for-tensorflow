import configparser
import importlib.util
import json
import os
import platform
import pip
import sys
import subprocess
import wget

def getConfig(url):
  filename = './local_config.json'
  if os.path.exists(filename) is False:
    wget.download(url, filename)
  with open(filename, 'r') as f:
    config = json.load(f)
    return config

def check_python():
  print("Check Python")

  itex_found = importlib.util.find_spec("intel_extension_for_tensorflow")
  if itex_found is None:
    exit("Please Install Intel(R) Extension for TensorFlow* first.\n")
  
  location = ''.join(itex_found.submodule_search_locations) + "/python/"
  sys.path.append(location)
  try:
    from version import __version__
    itex_version = __version__
  except Exception:
    print("Intel(R) Extension for TensorFlow* Version is Unknown.\n")
  python_major_version = sys.version_info.major
  python_minor_version = sys.version_info.minor
  python_micro_version = sys.version_info.micro

  if python_major_version < 2 :
    exit("Python2 is not supported, please install Python3!")
  elif python_minor_version < config['python_version']['min_python_version'][itex_version]:
    exit("Your Python version is too low, please upgrade to 3." + 
    str(config['python_version']['min_python_version'][itex_version]) + " or higher!")
  elif python_minor_version > config['python_version']['max_python_version'][itex_version]:
    exit("Your Python version is too high, please downgrade to 3." + 
    str(config['python_version']['max_python_version'][itex_version]) + " or lower!")
  print("\t Python " + str(python_major_version) + "." + str(python_minor_version) + "." + 
        str(python_micro_version) + " is Supported.")
  print("Check Python Passed\n")
  return itex_version

def check_os():
  print("Check OS")
  system_type = platform.system()
  if system_type != 'Linux':
    exit("We only Support Linux System\n")

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
    exit("Intel GPU Driver Does Not Support OS " + os_id + " : " + os_version + "\n Check OS failed. \n")
  print("Check OS Passed\n")
  return os_id, os_version

def check_tensorflow():
  print("Check Tensorflow")

  tf_found = importlib.util.find_spec("tensorflow")
  if tf_found is None:
    exit("Please Install TensorFlow first.\n")
  
  file = ''.join(tf_found.submodule_search_locations) + "/tools/pip_package/setup.py"
  cmd = "cat " + file + '|grep "_VERSION ="'
  res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  tf_version = str(res.stdout)[14:20]
  if tf_version < config['tensorflow_version']['min_tensorflow_version'][itex_version]:
    exit("Your Tensorflow version is too low, please upgrade to " + 
         str(config['tensorflow_version']['min_tensorflow_version'][itex_version]) + "!")
  print("\tTensorflow " + tf_version + " is installed.")
  print("Check Tensorflow Passed\n")

def check_driver():
  print("Check Intel GPU Driver")
  # tf_version = str(res.stdout)[14:20]
  if os_id == "ubuntu":
    cmd = "dpkg -s "
  elif os_id == "rhel":
    cmd = "yum info installed "
    print(res)
  elif os_id == "sles":
        cmd = "zypper se --installed-only "
  else:
    exit("Unsupported OS \n Check Intel GPU Driver Failed\n")

  for driver in config['intel_gpu_driver_list'][os_id]:
      if os.system(cmd + driver) != 0:
        exit("\tCheck Intel GPU Driver Failed\n")

  print("Check Intel GPU Driver Passsed\n")

def check_oneapi():
  print("Check OneAPI")
  cmd = 'LD_DEBUG=libs python -c "import intel_extension_for_tensorflow" 2>&1 |tee /tmp/log' 
  subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  for oneapi in config['oneapi_lib']:
    cmd = 'grep ' + oneapi + ' /tmp/log'
    res = os.system(cmd)
    if res != 0:
      exit("\tCan't find " + oneapi + "\n Check OneAPI Failed\n" )
    print("\t" + config['oneapi'][oneapi] + " is Installed.")
  print("Check OneAPI Passed\n")

def check_py_lib():
  print("Check Tensorflow Requirements\n")
  for lib in config['tf_requirements']:
    lib_found = importlib.util.find_spec(lib)
    if lib_found is None:
      print("\t" + lib + " should be installed.")
  print("Check Intel(R) Extension for TensorFlow* Requirements")
  for lib in config['itex_requirements']:
    lib_found = importlib.util.find_spec(lib)
    if lib_found is None:
      print("\t" + lib + " should be installed.")
  print("\n")

if __name__ == '__main__':
  print("\nCheck Environment for Intel(R) Extension for TensorFlow*...\n")
  url="https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/tools/python/config.json"
  configfile="./config.json"
  os_id=""
  config = getConfig(url)
  itex_version = check_python()
  os_id, os_version = check_os()
  check_tensorflow()
  check_driver()
  check_oneapi()
  check_py_lib()
