# Return the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")

def if_linux_x86_64(a, otherwise = []):
    return select({
        "//conditons:default": otherwise,
    })

def tf_copts(android_optimization_level_override = "-O2", is_external = False):
    # For compatibility reasons, android_optimization_level_override
    # is currently only being set for Android.
    # To clear this value, and allow the CROSSTOOL default
    # to be used, pass android_optimization_level_override=None
    return (
        [
            "-Wno-sign-compare",
            "-Wno-unknown-pragmas",
            # "-fno-exceptions", # TODO(itex): disable it first as we need expection in SE's dpcpp backend
            "-ftemplate-depth=900",
            "-msse3",
            "-pthread",
        ]
    )

def itex_xpu_library(name, srcs = [], hdrs = [], deps = [], *argc, **kwargs):
    kwargs["copts"] = kwargs.get("copts", []) + if_dpcpp(["-dpcpp_compile"])
    kwargs["linkopts"] = kwargs.get("linkopts", []) + if_dpcpp(["-link_stage"])
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        **kwargs
    )

def itex_xpu_binary(name, srcs = [], deps = [], *argc, **kwargs):
    kwargs["copts"] = kwargs.get("copts", []) + if_dpcpp(["-dpcpp_compile"])
    kwargs["linkopts"] = kwargs.get("linkopts", []) + if_dpcpp(["-link_stage"])
    native.cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        **kwargs
    )

def _get_transitive_headers(hdrs, deps):
    return depset(
        hdrs,
        transitive = [dep[CcInfo].compilation_context.headers for dep in deps],
    )

def _transitive_hdrs_impl(ctx):
    outputs = _get_transitive_headers([], ctx.attr.deps)
    return struct(files = outputs)

_transitive_hdrs = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_hdrs_impl,
)

def transitive_hdrs(name, deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"])

def cc_header_only_library(name, deps = [], includes = [], extra_deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.cc_library(
        name = name,
        srcs = [":" + name + "_gather"],
        hdrs = includes,
        deps = extra_deps,
        **kwargs
    )

def if_not_jax(if_true, if_false = []):
    """Shorthand for select()' on whether build .so for jax

    Returns `if_true` if graph compiler is used.
    graph compiler allows some accelerated kernel such as MHA in Bert
    """
    return select({
        "//itex:build_for_jax": if_false,
        "//conditions:default": if_true,
    })

def if_jax(if_true, if_false = []):
    """Shorthand for select()' on whether build .so for jax

    Returns `if_true` if graph compiler is used.
    graph compiler allows some accelerated kernel such as MHA in Bert
    """
    return select({
        "//itex:build_for_jax": if_true,
        "//conditions:default": if_false,
    })
