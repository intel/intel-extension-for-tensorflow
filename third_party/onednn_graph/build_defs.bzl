load("@bazel_skylib//lib:selects.bzl", "selects")

def if_llga_debug(if_true, if_false = []):
    """Shorthand for select()' on whether DEBUG mode LLGA is used.

    Returns `if_true` if DEBUG mode LLGA is used.
    DEBUG mode LLGA allows to dump LLGA graph and displays LLGA op verbose
    """
    return select({
        "@intel_extension_for_tensorflow//third_party/onednn_graph:build_with_llga_debug": if_true,
        "//conditions:default": if_false,
    })

def if_graph_compiler(if_true, if_false = []):
    """Shorthand for select()' on whether graph compiler and its dependency LLVM is built into ITEX

    Returns `if_true` if graph compiler is used.
    graph compiler allows some accelerated kernel such as MHA in Bert
    """
    return select({
        "@intel_extension_for_tensorflow//third_party/onednn_graph:build_with_graph_compiler": if_true,
        "//conditions:default": if_false,
    })

def onednn_graph_deps():
    """Shorthand for select() to pull in the correct set of OneDNN Graph library deps.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return selects.with_or({
        (str(Label("//third_party/build_option/dpcpp:build_with_dpcpp")), "@intel_extension_for_tensorflow//itex:gpu_build"): ["@onednn_graph//:onednn_graph_gpu"],
        "//conditions:default": ["@onednn_graph//:onednn_graph_cpu"],
    })
