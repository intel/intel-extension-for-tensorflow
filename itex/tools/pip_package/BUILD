# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

load(
    "//itex:itex.bzl",
    "transitive_hdrs",
)
load(
    "@intel_extension_for_tensorflow//third_party/onednn:build_defs.bzl",
    "if_cpu_onednn",
    "omp_deps",
)
load(
    "@local_config_dpcpp//dpcpp:build_defs.bzl",
    "if_dpcpp",
)
load(
    "@local_config_syslibs//:build_defs.bzl",
    "if_not_system_lib",
)

# This returns a list of headers of all public header libraries (e.g.,
# framework, lib), and all of the transitive dependencies of those
# public headers.  Not all of the headers returned by the filegroup
# are public (e.g., internal headers that are included by public
# headers), but the internal headers need to be packaged in the
# pip_package for the public headers to be properly included.
#
# Public headers are therefore defined by those that are both:
#
# 1) "publicly visible" as defined by bazel
# 2) Have documentation.
#
# This matches the policy of "public" for our python API.
transitive_hdrs(
    name = "included_headers",
    deps = [
        "//itex/core:protos_all_cc",
    ] + if_dpcpp([
        "//third_party/build_option/dpcpp:itex_gpu_header",
    ]),
)

COMMON_PIP_DEPS = [
    "MANIFEST.in",
    "README.md",
    "itex_setup.py",
    "lib_setup.py",
    "included_headers",
    "//itex/python:_pywrap_itex.so",
    "//itex/core:protos_all_py",
] + if_dpcpp([
    "//itex:libitex_gpu.so",
]) + if_cpu_onednn([
    # TODO(itex): need refine if_cpu_onednn
    "//itex:itex_cpu",
    "//itex:libitex_cpu_internal_avx512.so",
    "//itex:libitex_cpu_internal_avx2.so",
])

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = ["simple_console"] + COMMON_PIP_DEPS + omp_deps(),
)
