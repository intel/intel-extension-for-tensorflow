"""This file contains BUILD extensions for building llvm_openmp.
TODO(itex): Delete this and reuse a similar function in third_party/llvm
after the TF 2.4 branch cut has passed.
"""

load(
    "@intel_extension_for_tensorflow//itex/core/utils:build_config.bzl",
    "cc_binary",
)

def dict_add(*dictionaries):
    """Returns a new `dict` that has all the entries of the given dictionaries.

    If the same key is present in more than one of the input dictionaries, the
    last of them in the argument list overrides any earlier ones.

    Args:
      *dictionaries: Zero or more dictionaries to be added.

    Returns:
      A new `dict` that has all the entries of the given dictionaries.
    """
    result = {}
    for d in dictionaries:
        result.update(d)
    return result

# TODO(itex): Replace the following calls to cc_binary with cc_library.
# cc_library should be used for files that are not independently executed. Using
# cc_library results in linking errors. For e.g on Linux, the build fails
# with the following error message.
# ERROR: //tensorflow/BUILD:689:1: Linking of rule '//tensorflow:libtensorflow_framework.so.2.4.0' failed (Exit 1)
# /usr/bin/ld.gold: error: symbol GOMP_parallel_loop_nonmonotonic_guided has undefined version VERSION
# /usr/bin/ld.gold: error: symbol GOMP_parallel_start has undefined version GOMP_1.0
# /usr/bin/ld.gold: error: symbol GOMP_cancellation_point has undefined version GOMP_4.0
# /usr/bin/ld.gold: error: symbol omp_set_num_threads has undefined version OMP_1.0
# ......
# ......

# MacOS build has not been tested, however since the MacOS build of openmp
# uses the same configuration as Linux, the following should work.
def libiomp5_cc_binary(name, cppsources, srcdeps, common_includes):
    cc_binary(
        name = name,
        srcs = cppsources + srcdeps +
               [
                   #linux & macos specific files
                   "runtime/src/z_Linux_util.cpp",
                   "runtime/src/kmp_gsupport.cpp",
                   "runtime/src/z_Linux_asm.S",
               ],
        copts = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
        includes = common_includes,
        linkopts = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
        linkshared = True,
        additional_linker_inputs = [":ldscript"],
        win_def_file = ":generate_def",  # This will be ignored for non Windows builds
        visibility = ["//visibility:public"],
    )
