load("@bazel_skylib//lib:selects.bzl", "selects")

def if_cpu_onednn(if_true, if_false = []):
    """Returns `if_true` if oneDNN on CPU is used.

    Shorthand for select()'ing on whether we're building with 'config=mkl'.

    Returns a select statement which evaluates to if_true if we're building
    with oneDNN on CPU. Otherwise, the select statement evaluates to if_false.

    """
    return selects.with_or({
        ("//third_party/onednn:build_with_onednn", "@intel_extension_for_tensorflow//itex:cpu_build"): if_true,
        "//conditions:default": if_false,
    })

def omp_deps():
    """Returns llvm openmp runtime library

    """

    return selects.with_or({
        ("//third_party/onednn:onednn_v3_and_cpu", "@intel_extension_for_tensorflow//itex:cpu_build_with_onednn_v3"): ["//third_party/onednn:intel_binary_blob"],
        ("//third_party/onednn:onednn_v2_and_cpu", "@intel_extension_for_tensorflow//itex:cpu_build_with_onednn_v2"): ["//third_party/onednn_v2:intel_binary_blob"],
        "//conditions:default": [],
    })

def onednn_deps():
    """Shorthand for select() to pull in the correct set of oneDNN library deps.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return selects.with_or({
        ("//third_party/onednn:onednn_v3_and_gpu", "@intel_extension_for_tensorflow//itex:gpu_build_with_onednn_v3"): ["@onednn_gpu//:onednn_gpu"],
        ("//third_party/onednn:onednn_v2_and_gpu", "@intel_extension_for_tensorflow//itex:gpu_build_with_onednn_v2"): ["@onednn_gpu_v2//:onednn_gpu"],
        ("//third_party/onednn:onednn_v3_and_cpu", "@intel_extension_for_tensorflow//itex:cpu_build_with_onednn_v3"): ["@onednn_cpu//:onednn_cpu"],
        "//conditions:default": ["@onednn_cpu_v2//:onednn_cpu"],
    })

def if_llga_debug(if_true, if_false = []):
    """Shorthand for select()' on whether DEBUG mode LLGA is used.

    Returns `if_true` if DEBUG mode LLGA is used.
    DEBUG mode LLGA allows to dump LLGA graph and displays LLGA op verbose
    """
    return select({
        "@intel_extension_for_tensorflow//third_party/onednn:build_with_llga_debug": if_true,
        "//conditions:default": if_false,
    })

def if_graph_compiler(if_true, if_false = []):
    """Shorthand for select()' on whether graph compiler and its dependency LLVM is built into ITEX

    Returns `if_true` if graph compiler is used.
    graph compiler allows some accelerated kernel such as MHA in Bert
    """
    return select({
        "@intel_extension_for_tensorflow//third_party/onednn:build_with_graph_compiler": if_true,
        "//conditions:default": if_false,
    })

def onednn_graph_deps():
    """Shorthand for select() to pull in the correct set of OneDNN Graph library deps.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return selects.with_or({
        ("//third_party/onednn:onednn_v3_and_gpu", "@intel_extension_for_tensorflow//itex:gpu_build_with_onednn_v3"): ["@onednn_gpu//:onednn_graph_gpu"],
        ("//third_party/onednn:onednn_v2_and_gpu", "@intel_extension_for_tensorflow//itex:gpu_build_with_onednn_v2"): ["@onednn_graph//:onednn_graph_gpu"],
        ("//third_party/onednn:onednn_v3_and_cpu", "@intel_extension_for_tensorflow//itex:cpu_build_with_onednn_v3"): ["@onednn_cpu//:onednn_graph_cpu"],
        "//conditions:default": ["@onednn_graph//:onednn_graph_cpu"],
    })
