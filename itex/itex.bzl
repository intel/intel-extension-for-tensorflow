# Return the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")
load("@bazel_skylib//lib:selects.bzl", "selects")

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

def if_cpu_build(if_true, if_false = []):
    return select({
        "@intel_extension_for_tensorflow//itex:cpu_build": if_true,
        "//conditions:default": if_false,
    })

def if_cpu_avx512_build(if_true, if_false = []):
    return select({
        "@intel_extension_for_tensorflow//itex:cpu_avx512_build": if_true,
        "//conditions:default": if_false,
    })

def if_gpu_build(if_true, if_false = []):
    return select({
        "@intel_extension_for_tensorflow//itex:gpu_build": if_true,
        "//conditions:default": if_false,
    })

def if_gpu_backend(if_true, if_false = []):
    return selects.with_or({
        ("@local_config_dpcpp//dpcpp:using_dpcpp", "@intel_extension_for_tensorflow//itex:xpu_build"): if_true,
        "//conditions:default": if_false,
    })

def if_cpu_backend(if_true, if_false = []):
    return selects.with_or({
        ("@intel_extension_for_tensorflow//third_party/onednn:build_with_onednn", "@intel_extension_for_tensorflow//itex:xpu_build"): if_true,
        "//conditions:default": if_false,
    })

def avx_copts():
    # CPU default build opts: "-mavx", "-mavx2"
    # CPU avx512 build opts: "-mavx", "-mavx2", "-mavx512f", "-mavx512pf", "-mavx512cd", "-mavx512bw", "-march=skylake-avx512", "-mavx512dq"
    # CPU CC build opts: "-march=native"
    return (
        select({
            "@intel_extension_for_tensorflow//itex:cpu_build": [
                "-mavx",
                "-mavx2",
            ],
            "//conditions:default": [],
        }) + select({
            "@intel_extension_for_tensorflow//itex:cpu_avx512_build": [
                "-mavx512f",
                "-mavx512pf",
                "-mavx512cd",
                "-mavx512bw",
                "-march=skylake-avx512",
                "-mavx512dq",
            ],
            "//conditions:default": [],
        }) + select({
            "@intel_extension_for_tensorflow//itex:cpu_cc_build": [
                "-march=native",
            ],
            "//conditions:default": [],
        })
    )

def _copt_transition_impl(settings, attr):
    _ignore = settings
    return {"//itex:itex_build_target": attr.set_target}

copt_transition = transition(
    implementation = _copt_transition_impl,
    inputs = [],
    outputs = ["//itex:itex_build_target"],
)

def _transition_rule_impl(ctx):
    actual_binary = ctx.attr.actual_binary[0]
    outfile = ctx.actions.declare_file(ctx.label.name)
    cc_binary_outfile = actual_binary[DefaultInfo].files.to_list()[0]

    ctx.actions.run_shell(
        inputs = [cc_binary_outfile],
        outputs = [outfile],
        command = "cp %s %s" % (cc_binary_outfile.path, outfile.path),
    )

    return [
        DefaultInfo(
            files = depset([outfile]),
            data_runfiles = actual_binary[DefaultInfo].data_runfiles,
        ),
    ]

transition_rule = rule(
    implementation = _transition_rule_impl,
    attrs = {
        "set_target": attr.string(default = ""),
        "actual_binary": attr.label(cfg = copt_transition),
        "_whitelist_function_transition": attr.label(
            default = "@bazel_tools//tools/whitelists/function_transition_whitelist",
        ),
    },
)

def cc_binary(name, set_target = None, srcs = [], deps = [], *argc, **kwargs):
    cc_binary_name = name.replace("lib", "").replace(".so", "")
    transition_rule(
        name = name,
        actual_binary = ":%s" % cc_binary_name,
        set_target = set_target,
    )
    native.cc_binary(
        name = cc_binary_name,
        srcs = srcs,
        deps = deps,
        **kwargs
    )

def cc_library(name, srcs = [], deps = [], *argc, **kwargs):
    kwargs["copts"] = kwargs.get("copts", []) + if_cpu_build(["-DINTEL_CPU_ONLY", "-mfma", "-O3"]) + avx_copts() + if_gpu_build(["-DINTEL_GPU_ONLY"])
    kwargs["linkopts"] = kwargs.get("linkopts", []) + if_gpu_build(["-DINTEL_GPU_ONLY"])
    native.cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        **kwargs
    )

def itex_xpu_library(name, srcs = [], hdrs = [], deps = [], *argc, **kwargs):
    kwargs["copts"] = kwargs.get("copts", []) + if_dpcpp(["-dpcpp_compile"]) + if_cpu_build(["-DINTEL_CPU_ONLY", "-mfma", "-O3"]) + avx_copts() + if_gpu_build(["-DINTEL_GPU_ONLY"])
    kwargs["linkopts"] = kwargs.get("linkopts", []) + if_dpcpp(["-link_stage"]) + if_gpu_build(["-DINTEL_GPU_ONLY"])
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        **kwargs
    )

def itex_xpu_binary(name, set_target = None, srcs = [], deps = [], *argc, **kwargs):
    xpu_binary_name = name.replace("lib", "").replace(".so", "")
    transition_rule(
        name = name,
        actual_binary = ":%s" % xpu_binary_name,
        set_target = set_target,
    )
    kwargs["copts"] = kwargs.get("copts", []) + if_dpcpp(["-dpcpp_compile"])
    kwargs["linkopts"] = kwargs.get("linkopts", []) + if_dpcpp(["-link_stage"])
    native.cc_binary(
        name = xpu_binary_name,
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
