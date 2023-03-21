'''
lib_setup.py file to build wheel for Intel® Extension for TensorFlow lib*
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import sys

from datetime import date
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'intel_extension_for_tensorflow/python')) # pylint: disable=line-too-long
from version import __version__


# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update itex/itex.bzl
_VERSION = __version__

REQUIRED_PACKAGES = []

if sys.byteorder == 'little':
  # grpcio does not build correctly on big-endian machines due to lack of
  # BoringSSL support.
  # See https://github.com/tensorflow/tensorflow/issues/17882.
  REQUIRED_PACKAGES.append('grpcio >= 1.8.6')

project_name = 'intel_extension_for_tensorflow_lib'
DEV_VERSION_SUFFIX = ""
if "--weekly_build" in sys.argv:
        today_number = date.today().strftime("%Y%m%d")
        DEV_VERSION_SUFFIX = ".dev" + today_number
        sys.argv.remove("--weekly_build")
        project_name = "itex_lib_weekly"
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1] + "_lib"
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)
REQUIRED_PACKAGES.append('wheel')
CONSOLE_SCRIPTS = []


_ext_path = 'intel_extension_for_tensorflow'
_ext_lib_path = 'intel_extension_for_tensorflow_lib'
_plugin_path = 'tensorflow-plugins'

# Get this lib is CPU or GPU
filenames = os.listdir(_plugin_path)
is_cpu = False
is_gpu = False
for filename in filenames:
  if "cpu" in filename:
    is_cpu = True
  if "gpu" in filename:
    is_gpu = True
if is_cpu and not is_gpu:
  _VERSION = _VERSION + ".0"
elif not is_cpu and is_gpu:
  _VERSION = _VERSION + ".1"
elif is_cpu and is_gpu:
  raise Exception("This version does not yet support both CPU and GPU.")
else:
  raise Exception("There are no .so files in the folder of \
                   tensorflow-plugins, please check it.")


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True

def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

matches = []
for path in so_lib_paths:
  matches.extend(
      ['../' + x for x in find_files('*', path) if '.py' not in x]
  )

env_check_tool = []
if is_gpu:
  env_check_tool = ['tools/*']

long_description = """# Intel® Extension for Tensorflow* library

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://pypi.org/project/intel-extension-for-tensorflow)
[![version](https://img.shields.io/badge/release-1.0.0-green)](https://github.com/intel/intel-extension-for-tensorflow/releases)

Intel® Extension for Tensorflow* library is the support library for Intel® Extension for Tensorflow*(https://pypi.org/project/intel-extension-for-tensorflow/). While Intel® Extension for Tensorflow* itself is a pure Python package, Intel® Extension for Tensorflow* library contains the binary (C/C++) parts of the library, including Python bindings, Intel XPU(GPU, CPU, etc) devices support.

Documentation: [**Intel® Extension for TensorFlow\* online document website**](https://intel.github.io/intel-extension-for-tensorflow/).

## Security
See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html) for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](https://intel.github.io/intel-extension-for-tensorflow/latest/SECURITY.html)
"""

class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)  # pylint: disable=assignment-from-no-return
    self.install_headers = os.path.join(self.install_platlib, \
                           'intel_extension_for_tensorflow',
                                        'include')
    self.install_lib = self.install_platlib
    return ret

setup(
    name=project_name,
    version=_VERSION.replace('-', '') + DEV_VERSION_SUFFIX,
    description='Intel® Extension for Tensorflow* library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # pylint: disable=line-too-long
    url='https://github.com/intel/intel-extension-for-tensorflow',
    download_url='https://github.com/intel/intel-extension-for-tensorflow/tags',
    project_urls={
        "Bug Tracker": "https://github.com/intel/intel-extension-for-tensorflow/issues",
    },
    # pylint: enable=line-too-long
    author='Intel Corporation',
    author_email='itex.maintainers@intel.com',
    # Contained modules and scripts.
    packages=[_ext_path, _ext_lib_path, _plugin_path],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # Add in any packaged data.
    package_data={
        _ext_path: [
            'python/*.so',
            'libitex_common.so'
        ] + matches + env_check_tool,
        _plugin_path: [
            '*'
        ],
        _ext_lib_path: [
            '__init__.py',
            '__main__.py',
            "third-party-programs/*",
        ],
    },
    exclude_package_data={
        'intel_extension_for_tensorflow': ['tools']
    },
    python_requires='>=3.7',
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='Intel® Extension for Tensorflow*',
        cmdclass={
            'install': InstallCommand,
        },
)
