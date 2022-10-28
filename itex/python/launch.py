#   Copyright (c) 2022 Intel Corporation
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ============================================================================

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import platform
import subprocess
import os
from os.path import expanduser
import re
import glob
from argparse import ArgumentParser, REMAINDER
from argparse import RawTextHelpFormatter
import logging
from datetime import datetime
import numpy as np

format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)


class CPUinfo():
  '''
  Get CPU inforamation, such as cores list and NUMA information.
  '''

  def __init__(self):

    self.cpuinfo = []
    if platform.system() == "Linux":
      args = ["lscpu", "--parse=CPU,Core,Socket,Node"]
      lscpu_info = subprocess.check_output(
          args, universal_newlines=True).split("\n")

      # Get information about  cpu, core, socket and node
      for line in lscpu_info:
        pattern = r"^([\d]+,[\d]+,[\d]+,[\d]?)"
        regex_out = re.search(pattern, line)
        if regex_out:
          self.cpuinfo.append(regex_out.group(1).strip().split(","))
      assert len(self.cpuinfo) > 0, "cpuinfo is empty"
      self.get_socket_info()
    else:
      raise RuntimeError(
        "{} platform is not supported!!!".format(platform.system()))

  def get_socket_info(self):
    idx_active = 3
    if self.cpuinfo[0][idx_active] == '':
      idx_active = 2
    self.nodes = int(max([line[idx_active] for line in self.cpuinfo])) + 1
    self.node_physical_cores = []  # node_id is index
    self.node_logical_cores = []   # node_id is index
    self.physical_core_node_map = {}  # phyical core to numa node id
    self.logical_core_node_map = {}   # logical core to numa node id

    for node_id in range(self.nodes):
      cur_node_physical_core = []
      cur_node_logical_core = []
      for line in self.cpuinfo:
        nid = line[idx_active] if line[idx_active] != '' else '0'
        if node_id == int(nid):
          if int(line[1]) not in cur_node_physical_core:
            cur_node_physical_core.append(int(line[1]))
            self.physical_core_node_map[int(
                line[1])] = int(node_id)
          cur_node_logical_core.append(int(line[0]))
          self.logical_core_node_map[int(line[0])] = int(node_id)
      self.node_physical_cores.append(cur_node_physical_core)
      self.node_logical_cores.append(cur_node_logical_core)

  def node_nums(self):
    return self.nodes

  def physical_core_nums(self):
    return len(self.node_physical_cores) * len(self.node_physical_cores[0])

  def logical_core_nums(self):
    return len(self.node_logical_cores) * len(self.node_logical_cores[0])

  def get_node_physical_cores(self, node_id):
    if node_id < 0 or node_id > self.nodes - 1:
      logger.error("Invalid node id")
    return self.node_physical_cores[node_id]

  def get_node_logical_cores(self, node_id):
    if node_id < 0 or node_id > self.nodes - 1:
      logger.error("Invalid node id")
    return self.node_logical_cores[node_id]

  def get_all_physical_cores(self):
    return np.array(self.node_physical_cores).flatten().tolist()

  def get_all_logical_cores(self):
    return np.array(self.node_logical_cores).flatten().tolist()

  def numa_aware_check(self, core_list):
    '''
    Check whether all cores in core_list are in the same NUMA node.
    Cross NUMA will reduce perforamnce.
    We strongly advice to not use cores on different nodes.
    '''
    cores_numa_map = self.logical_core_node_map
    numa_ids = []
    for core in core_list:
      numa_id = cores_numa_map[core]
      if not numa_id in numa_ids:
        numa_ids.append(numa_id)
    if len(numa_ids) > 1:
      logger.warning("Numa Aware: cores:{} on different NUMA nodes:{}".format(
          str(core_list), str(numa_ids)))
    if len(numa_ids) == 0:
      logger.error(
          "invalid number of NUMA nodes; please make sure numa_ids >= 1")
      exit(-1)
    return numa_ids


class Launcher():
  r"""
   Base class for launcher
  """

  def __init__(self):
    self.cpuinfo = CPUinfo()

  def launch(self, args):
    pass

  def add_lib_preload(self, lib_type=None):
    '''
    Enale TCMalloc/JeMalloc/intel OpenMP
    '''
    library_paths = []
    if "CONDA_PREFIX" in os.environ:
      library_paths.append(os.environ["CONDA_PREFIX"] + "/lib/")
    if "VIRTUAL_ENV" in os.environ:
      library_paths.append(os.environ["VIRTUAL_ENV"] + "/lib/")

    library_paths += ["{}/.local/lib/".format(expanduser("~")),
                      "/usr/local/lib/", "/usr/local/lib64/",
                      "/usr/lib/", "/usr/lib64/"]

    lib_find = False
    lib_set = False
    for item in os.getenv("LD_PRELOAD", "").split(":"):
      if item.endswith('lib{}.so'.format(lib_type)):
        lib_set = True
        break
    if not lib_set:
      for lib_path in library_paths:
        library_file = lib_path + "lib" + lib_type + ".so"
        matches = glob.glob(library_file)
        if len(matches) > 0:
          if "LD_PRELOAD" in os.environ:
            os.environ["LD_PRELOAD"] = matches[0] + \
                ":" + os.environ["LD_PRELOAD"]
          else:
            os.environ["LD_PRELOAD"] = matches[0]
          lib_find = True
          break
    return lib_set or lib_find

  def is_numactl_available(self):
    numactl_available = False
    cmd = ["numactl", "-C", "0", "-m", "0", "ls"]
    r = subprocess.run(cmd, env=os.environ, stdout=subprocess.DEVNULL)
    if r.returncode == 0:
      numactl_available = True
    return numactl_available

  def set_memory_allocator(self, enable_tcmalloc=True,
                           enable_jemalloc=False, use_default_allocator=False):
    '''
    Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
    '''
    if enable_tcmalloc and enable_jemalloc:
      logger.error(
          "Unable to enable TCMalloc and JEMalloc at the same time")
      exit(-1)

    if enable_tcmalloc:
      find_tc = self.add_lib_preload(lib_type="tcmalloc")
      if not find_tc:
        logger.warning("Unable to find the {} library file lib{}.so"
                       " in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                       " or /.local/lib/ or /usr/local/lib/ or"
                       " /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                       "{}/.local/lib/ so the LD_PRELOAD environment variable"
                       " will not be set. You can use "
                       "'conda install -c conda-forge gperftools' "
                       "to install tcmalloc"
                       .format("TCmalloc", "tcmalloc", expanduser("~")))
      else:
        logger.info("Use TCMalloc memory allocator")

    elif enable_jemalloc:
      find_je = self.add_lib_preload(lib_type="jemalloc")
      if not find_je:
        logger.warning("Unable to find the {} library file lib{}.so"
                       " in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                       " or /.local/lib/ or /usr/local/lib/ or"
                       " /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                       "{}/.local/lib/ so the LD_PRELOAD environment variable"
                       " will not be set. You can use "
                       "'conda install -c conda-forge jemalloc'"
                       " to install jemalloc"
                       .format("JeMalloc", "jemalloc", expanduser("~")))
      else:
        logger.info("Use JeMalloc memory allocator")
        self.set_env(
            'MALLOC_CONF',
            "oversize_threshold:1,background_thread:true,metadata_thp:auto")

    elif use_default_allocator:
      pass

    else:
      find_tc = self.add_lib_preload(lib_type="tcmalloc")
      if find_tc:
        logger.info("Use TCMalloc memory allocator")
        return
      find_je = self.add_lib_preload(lib_type="jemalloc")
      if find_je:
        logger.info("Use JeMalloc memory allocator")
        return
      logger.warning("Neither TCMalloc nor JeMalloc is found in"
                     " $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                     " or /.local/lib/ or /usr/local/lib/ or "
                     "/usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                     "{}/.local/lib/ so the LD_PRELOAD environment "
                     "variable will not be set. This may drop the performance."
                     .format(expanduser("~")))

  def logger_env(self, env_name=""):
    if env_name in os.environ:
      logger.info("{}={}".format(env_name, os.environ[env_name]))

  def set_env(self, env_name, env_value=None):
    if not env_value:
      logger.warning("{} is None".format(env_name))
    if env_name not in os.environ:
      os.environ[env_name] = env_value
    elif os.environ[env_name] != env_value:
      logger.warning("{} in environment variable is {} "
            "while the value you set is {}"
            .format(env_name, os.environ[env_name], env_value))
    self.logger_env(env_name)

  # set_kmp_affinity is used to control whether to set KMP_AFFINITY or not.
  # In scenario that use all cores on all nodes, including logical cores,
  # setting KMP_AFFINITY disables logical cores. In this case, KMP_AFFINITY
  # should not be set.
  def set_multi_thread_and_allocator(self, ncore_per_instance, num_inter,
                                    num_intra, set_kmp_affinity=True,
                                    enable_tcmalloc=True, enable_jemalloc=False,
                                    use_default_allocator=False):
    '''
    Set multi-thread configuration and enable LLVM openMP and TCMalloc/JeMalloc.
    '''
    self.set_memory_allocator(
        enable_tcmalloc, enable_jemalloc, use_default_allocator)
    self.set_env("OMP_NUM_THREADS", str(ncore_per_instance))
    if set_kmp_affinity:
      if len(self.cpuinfo.get_node_logical_cores(0)) > len(
              self.cpuinfo.get_node_physical_cores(0)):
        # HT is on
        self.set_env("KMP_AFFINITY",
                     "granularity=fine,verbose,compact,1,0")
      else:
        # HT is off
        self.set_env("KMP_AFFINITY",
                     "granularity=fine,verbose,compact,")

    self.set_env("KMP_BLOCKTIME", "1")
    if num_inter is None:
      self.set_env("TF_NUM_INTEROP_THREADS", "1")
    else:
      try:
        num = int(num_inter)
        assert num >= -1
      except BaseException:
        logger.error(
            "tf_num_interop_threads should be an integer >= -1, "
            "but input is {}.".format(num_inter))
        exit(-1)
      self.set_env("TF_NUM_INTEROP_THREADS", num_inter)
    if num_intra is None:
      self.set_env("TF_NUM_INTRAOP_THREADS", str(ncore_per_instance))
    else:
      try:
        num = int(num_intra)
        assert num >= 0
      except BaseException:
        logger.error(
            "tf_num_intraop_threads should be an integer >= 0, "
            "but input is {}.".format(num_intra))
        exit(-1)
      self.set_env("TF_NUM_INTRAOP_THREADS", num_intra)

  def set_itex(self, amp=False, enable_layout=False):
    self.set_env("TF_ENABLE_ONEDNN_OPTS", "1")
    if amp:
      self.set_env("ITEX_AUTO_MIXED_PRECISION", "1")
    if enable_layout:
      self.set_env("ITEX_LAYOUT_OPT", "1")
    else:
      self.set_env("ITEX_LAYOUT_OPT", "0")


class MultiInstanceLauncher(Launcher):
  r"""
   Launcher for single instance and multi-instance
   """

  def launch(self, args):
    processes = []
    cores = []
    set_kmp_affinity = True
    enable_taskset = False
    if args.core_list:  # user specify what cores will be used by params
      cores = [int(x) for x in args.core_list.split(",")]
      if args.ncore_per_instance == -1:
        logger.error(
            "please specify the '--ncore_per_instance' "
            "if you have pass the --core_list params")
        exit(-1)
      elif args.ninstances > 1 and args.ncore_per_instance * args.ninstances < len(cores):
        logger.warning("only first {} cores will be used, "
        "but you specify {} cores in core_list".format(
            args.ncore_per_instance * args.ninstances, len(cores)))
      else:
        args.ninstances = len(cores) // args.ncore_per_instance

    else:
      if args.use_logical_core:
        if args.node_id != -1:
          cores = self.cpuinfo.get_node_logical_cores(args.node_id)
        else:
          cores = self.cpuinfo.get_all_logical_cores()
          # When using all cores on all nodes, including logical cores,
          # setting KMP_AFFINITY disables logical cores.
          # Thus, KMP_AFFINITY should not be set.
          # set_kmp_affinity = False
      else:
        if args.node_id != -1:
          cores = self.cpuinfo.get_node_physical_cores(args.node_id)
        else:
          cores = self.cpuinfo.get_all_physical_cores()

      def skip_cores(cores):
        ncore_per_node = len(self.cpuinfo.node_physical_cores[0])
        num_leftover_cores = ncore_per_node % args.ncore_per_instance
        if args.ncore_per_instance > ncore_per_node:
          # too many ncore_per_instance to skip cross-node cores
          logger.warning("there are {} core(s) per socket, "
                         "but you specify {} ncore_per_instance "
                         "and skip_cross_node_cores. "
                         "Please make sure --ncore_per_instance < core(s) "
                         "per socket".format(
              ncore_per_node, args.ncore_per_instance))
          exit(-1)
        elif num_leftover_cores == 0:
          # aren't any cross-node cores
          logger.info(
              '--skip_cross_node_cores is set, but there are no cross-node cores.')
          args.ninstances = len(cores) // args.ncore_per_instance
        else:
          # skip cross-node cores
          if args.ninstances != -1:
            logger.warning(
                "--skip_cross_node_cores is exclusive to --ninstances. "
                "--ninstances won\'t take effect even if it is set explicitly.")
          i = 1
          leftover_cores = set()
          while ncore_per_node * i <= len(cores):
            leftover_cores.update(
                cores[ncore_per_node * i - num_leftover_cores: ncore_per_node * i])
            i += 1
          cores = list(set(cores) - leftover_cores)
          assert len(cores) % args.ncore_per_instance == 0
          args.ninstances = len(cores) // args.ncore_per_instance
          return cores
      if not args.multi_instance and args.ninstances == - \
              1 and args.ncore_per_instance == -1:
        args.ninstances = 1
        args.ncore_per_instance = len(cores)
      elif args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
        args.throughput_mode = True
      elif args.ncore_per_instance == -1 and args.ninstances != -1:
        if args.ninstances > len(cores):
          logger.error("there are {} total cores but you specify {} ninstances; "
              "please make sure ninstances <= total_cores)".format(
              len(cores), args.ninstances))
          exit(-1)
        else:
          args.ncore_per_instance = len(cores) // args.ninstances
      elif args.ncore_per_instance != -1 and args.ninstances == -1:
        if not args.skip_cross_node_cores:
          args.ninstances = len(cores) // args.ncore_per_instance
        else:
          cores = skip_cores(cores)
      elif args.ncore_per_instance != -1 and args.ninstances != -1 and args.skip_cross_node_cores:
        cores = skip_cores(cores)
      else:
        if args.ninstances * args.ncore_per_instance > len(cores):
          logger.error(
              "Please make sure ninstances * ncore_per_instance <= total_cores")
          exit(-1)
      if args.latency_mode:
        logger.warning(
            "--latency_mode is exclusive to --ninstances, --ncore_per_instance, "
            "--node_id and --use_logical_core. "
            "They won\'t take effect even if they are set explicitly.")
        args.ncore_per_instance = 4
        cores = self.cpuinfo.get_all_physical_cores()
        args.ninstances = len(cores) // args.ncore_per_instance

      if args.throughput_mode:
        logger.warning(
            "--throughput_mode is exclusive to --ninstances, --ncore_per_instance, "
            "--node_id and --use_logical_core. "
            "They won\'t take effect even if they are set explicitly.")
        args.ninstances = self.cpuinfo.node_nums()
        cores = self.cpuinfo.get_all_physical_cores()
        args.ncore_per_instance = len(cores) // args.ninstances

    if args.ninstances > 1 and args.instance_idx != -1:
      logger.info("assigning {} cores for instance {}".format(
          args.ncore_per_instance, args.instance_idx))

    if not args.disable_numactl:
      numactl_available = self.is_numactl_available()
      if not numactl_available:
        if not args.disable_taskset:
          logger.warning(
              "Core binding with numactl is not available. "
              "Disabling numactl and using taskset instead. "
              "This may affect performance in multi-socket system; "
              "please use numactl if memory binding is needed.")
          args.disable_numactl = True
          enable_taskset = True
        else:
          logger.warning(
              "Core binding with numactl is not available, "
              "and --disable_taskset is set. "
              "Please unset --disable_taskset to use taskset insetad of numactl.")
          exit(-1)

    if not args.disable_taskset:
      enable_taskset = True

    self.set_multi_thread_and_allocator(args.ncore_per_instance,
                                        args.tf_num_interop_threads,
                                        args.tf_num_intraop_threads,
                                        set_kmp_affinity,
                                        args.enable_tcmalloc,
                                        args.enable_jemalloc,
                                        args.use_default_allocator)
    self.set_itex(args.enable_itex_amp, args.enable_itex_layout_opt)
    os.environ["LAUNCH_CMD"] = "#"
    for i in range(args.ninstances):
      cmd = []
      cur_process_cores = ""
      if not args.disable_numactl or enable_taskset:
        if not args.disable_numactl:
          cmd = ["numactl"]
        elif enable_taskset:
          cmd = ["taskset"]

        cores = sorted(cores)
        if args.instance_idx == -1:  # sequentially assign ncores_per_instance to ninstances
          core_list = cores[i * args.ncore_per_instance: (
              i + 1) * args.ncore_per_instance]
        else:  # assign ncores_per_instance from instance_idx
          core_list = cores[args.instance_idx * args.ncore_per_instance: (
              args.instance_idx + 1) * args.ncore_per_instance]

        core_ranges = []
        for core in core_list:
          if len(core_ranges) == 0:
            range_elem = {'start': core, 'end': core}
            core_ranges.append(range_elem)
          else:
            if core - core_ranges[-1]['end'] == 1:
              core_ranges[-1]['end'] = core
            else:
              range_elem = {'start': core, 'end': core}
              core_ranges.append(range_elem)
        for r in core_ranges:
          cur_process_cores = cur_process_cores + \
              "{}-{},".format(r['start'], r['end'])
        cur_process_cores = cur_process_cores[:-1]
        if not args.disable_numactl:
          numa_params = "--localalloc "
          numa_params += "-C {} ".format(cur_process_cores)
          cmd.extend(numa_params.split())
        elif enable_taskset:
          taskset_params = "-c {}".format(cur_process_cores)
          cmd.extend(taskset_params.split())

      with_python = not args.no_python
      if with_python:
        cmd.append(sys.executable)
        cmd.append("-u")
      if args.module:
        cmd.append("-m")
      cmd.append(args.program)
      log_name = args.log_file_prefix + \
          "_instance_{}_cores_".format(
              i) + cur_process_cores.replace(',', '_') + ".log"
      log_name = os.path.join(args.log_path, log_name)
      cmd.extend(args.program_args)
      os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
      cmd_s = " ".join(cmd)
      if args.log_path:
        cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_name)
      logger.info(cmd_s)
      if not args.disable_numactl:
        process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
      elif enable_taskset:
        process = subprocess.Popen(cmd, env=os.environ)
      processes.append(process)

      if args.instance_idx != -1:  # launches single instance, instance_idx, only
        break
    os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]
    for process in processes:
      process.wait()
      if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode, cmd=cmd_s)


def add_itex_params(parser):

  group = parser.add_argument_group("ITEX Parameters")
  # ITEX control
  group.add_argument("--enable_itex_amp", action='store_true', default=False,
                     help="Enable ITEX AMP")
  group.add_argument("--enable_itex_layout_opt", action='store_true', default=False,
                     help="Enable ITEX layout opt")


def add_memory_allocator_params(parser):

  group = parser.add_argument_group("Memory Allocator Parameters")
  # allocator control
  group.add_argument("--enable_tcmalloc", action='store_true', default=False,
                     help="Enable tcmalloc allocator")
  group.add_argument("--enable_jemalloc", action='store_true', default=False,
                     help="Enable jemalloc allocator")
  group.add_argument("--use_default_allocator", action='store_true', default=False,
                     help="Use default memory allocator")


def add_multi_instance_params(parser):

  group = parser.add_argument_group("Multi-instance Parameters")
  # multi-instance control
  group.add_argument("--ncore_per_instance", metavar='\b', default=-1, type=int,
                     help="Cores per instance")
  group.add_argument("--skip_cross_node_cores", action='store_true', default=False,
                     help="If specified --ncore_per_instance, skips cross-node cores.")
  group.add_argument("--ninstances", metavar='\b', default=-1, type=int,
                     help="For multi-instance, you should give the cores number you used for per instance.")
  group.add_argument("--instance_idx", metavar='\b', default="-1", type=int,
                     help="Specify instance index to assign ncores_per_instance for instance_idx; otherwise ncore_per_instance will be assigned sequentially to ninstances. Please refer to https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/tree/master/docs/guide/launch.md")
  group.add_argument("--latency_mode", action='store_true', default=False,
                     help="By detault 4 core per instance and use all physical cores")
  group.add_argument("--throughput_mode", action='store_true', default=False,
                     help="By default one instance per node and use all physical cores")
  group.add_argument("--node_id", metavar='\b', default=-1, type=int,
                     help="node id for multi-instance, by default all nodes will be used")
  group.add_argument("--use_logical_core", action='store_true', default=False,
                     help="Whether only use physical cores")
  group.add_argument("--disable_numactl", action='store_true', default=False,
                     help="Disable numactl")
  group.add_argument("--disable_taskset", action='store_true', default=False,
                     help="Disable taskset")
  group.add_argument("--core_list", metavar='\b', default=None, type=str,
                     help="Specify the core list as 'core_id, core_id, ....', otherwise, all the cores will be used.")
  group.add_argument("--tf_num_interop_threads", metavar='\b', default=None, type=str,
                     help="Set TF_NUM_INTEROP_THREADS, by default it is 1.")
  group.add_argument("--tf_num_intraop_threads", metavar='\b', default=None, type=str,
                     help="Set TF_NUM_INTRAOP_THREADS, by default it equals to number of cores per instance.")
  group.add_argument("--log_path", metavar='\b', default="", type=str,
                     help="The log file directory. Default path is '', which means disable logging to files.")
  group.add_argument("--log_file_prefix", metavar='\b', default="run", type=str,
                     help="log file prefix")


def parse_args():
  """
  Helper function parsing the command line options
  @retval ArgumentParser
  """
  parser = ArgumentParser(description="This is a script for launching Tensorflow training and inference on Intel Xeon CPU "
                                      "with optimal configurations. Now, single instance inference/training, multi-instance "
                                      "inference/training are enabled. "
                                      "To get the peak performance on Intel Xeon CPU, the script optimizes the configuration "
                                      "of thread and memory management. For thread management, the script configures thread "
                                      "affinity and the preload of Intel OMP library. For memory management, it configures "
                                      "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
                                      "\n################################# Basic usage ############################# \n"
                                      "\n 1. single instance\n"
                                      "\n   >>> python -m intel_extension_for_tensorflow.python.launch python_script args \n"
                                      "\n2. multi-instance \n"
                                      "\n    >>> python -m intel_extension_for_tensorflow.python.launch --ninstances xxx --ncore_per_instance xx python_script args\n"
                                      "\n############################################################################# \n",
                                      formatter_class=RawTextHelpFormatter)

  parser.add_argument("--multi_instance", action='store_true', default=False,
                      help="Enable multi-instance, by default one instance per socket")

  parser.add_argument("-m", "--module", default=False, action="store_true",
                      help="Changes each process to interpret the launch script "
                           "as a python module, executing with the same behavior as"
                           "'python -m'.")

  parser.add_argument("--no_python", default=False, action="store_true",
                      help="Do not prepend the --program script with \"python\" - just exec "
                           "it directly. Useful when the script is not a Python script.")

  add_memory_allocator_params(parser)
  add_itex_params(parser)
  add_multi_instance_params(parser)
  # positional
  parser.add_argument("program", type=str,
                      help="The full path to the proram/script to be launched. "
                           "followed by all the arguments for the script")

  # rest from the training program
  parser.add_argument('program_args', nargs=REMAINDER)
  return parser.parse_args()


def main():
  env_before = set(os.environ.keys())
  if platform.system() == "Windows":
    raise RuntimeError("Windows platform is not supported!!!")

  args = parse_args()
  if args.log_path:
    path = os.path.dirname(args.log_path if args.log_path.endswith(
        '/') else args.log_path + '/')
    if not os.path.exists(path):
      os.makedirs(path)
    args.log_path = path

    args.log_file_prefix = '{}_{}'.format(
        args.log_file_prefix, datetime.now().strftime("%Y%m%d%H%M%S"))
    fileHandler = logging.FileHandler(
        "{0}/{1}_instances.log".format(args.log_path, args.log_file_prefix))
    logFormatter = logging.Formatter(format_str)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

  if args.latency_mode and args.throughput_mode:
    raise RuntimeError(
        "Either args.latency_mode or args.throughput_mode should be set")

  if not args.no_python and not args.program.endswith(".py"):
    logger.error(
        "For non Python script, you should use '--no_python' parameter.")
    exit()

  # Verify LD_PRELOAD
  if "LD_PRELOAD" in os.environ:
    lst_valid = []
    tmp_ldpreload = os.environ["LD_PRELOAD"]
    for item in tmp_ldpreload.split(":"):
      if item != "":
        matches = glob.glob(item)
        if len(matches) > 0:
          lst_valid.append(item)
        else:
          logger.warning(
              "{} doesn't exist. Removing it from LD_PRELOAD.".format(item))
    if len(lst_valid) > 0:
      os.environ["LD_PRELOAD"] = ":".join(lst_valid)
    else:
      os.environ["LD_PRELOAD"] = ""

  launcher = MultiInstanceLauncher()
  launcher.launch(args)
  for x in sorted(set(os.environ.keys()) - env_before):
    logger.debug('{0}={1}'.format(x, os.environ[x]))


if __name__ == "__main__":
  main()
