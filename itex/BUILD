package(default_visibility = ["//visibility:public"])

load("//itex:itex.bzl", "cc_binary", "itex_xpu_binary", "itex_xpu_library", "tf_copts")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")
load("@bazel_skylib//lib:selects.bzl", "selects")

string_flag(
    name = "itex_build_target",
    build_setting_default = "",
    values = [
        "cpu_avx2_backend",
        "cpu_avx512_backend",
        "cpu_cc_backend",
        "gpu_orig_backend",
        "gpu_cc_backend",
        "xla_extension",
        "dummy",
        "",
    ],
)

config_setting(
    name = "cpu_avx2_build",
    flag_values = {":itex_build_target": "cpu_avx2_backend"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cpu_avx512_build",
    flag_values = {":itex_build_target": "cpu_avx512_backend"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cpu_cc_build",
    flag_values = {":itex_build_target": "cpu_cc_backend"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "cpu_build",
    match_any = [
        ":cpu_avx2_build",
        ":cpu_avx512_build",
        ":cpu_cc_build",
    ],
)

config_setting(
    name = "gpu_orig_build",
    flag_values = {":itex_build_target": "gpu_orig_backend"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "gpu_cc_build",
    flag_values = {":itex_build_target": "gpu_cc_backend"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "gpu_build",
    match_any = [
        ":gpu_orig_build",
        ":gpu_cc_build",
    ],
)

selects.config_setting_group(
    name = "if_using_dpcpp",
    match_any = [
        "@local_config_dpcpp//dpcpp:using_dpcpp",
        ":gpu_build",
    ],
)

selects.config_setting_group(
    name = "gpu_build_with_onednn_v3",
    match_all = [
        ":gpu_build",
        "//third_party/onednn:onednn_v3",
    ],
)

selects.config_setting_group(
    name = "gpu_build_with_onednn_v2",
    match_all = [
        ":gpu_build",
        "//third_party/onednn:onednn_v2",
    ],
)

selects.config_setting_group(
    name = "cpu_build_with_onednn_v3",
    match_all = [
        ":cpu_build",
        "//third_party/onednn:onednn_v3",
    ],
)

selects.config_setting_group(
    name = "cpu_build_with_onednn_v2",
    match_all = [
        ":cpu_build",
        "//third_party/onednn:onednn_v2",
    ],
)

config_setting(
    name = "tf_version_2_12",
    define_values = {"tf_main_version": "2.13"},
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libitex_gpu.so",
    srcs = [
        "//itex/core/wrapper:itex_gpu_wrapper",
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow"],
    linkshared = True,
    set_target = "dummy",
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:device_backend_util_hdr",
        "//itex/core/kernels:libitex_common",
        "//itex/core/utils:env_var",
        "//itex/core/utils:logging",
    ],
)

itex_xpu_binary(
    name = "libitex_gpu_internal.so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow",
        "-DINTEL_GPU_ONLY",
    ],
    linkshared = True,
    set_target = "gpu_orig_backend",
    visibility = ["//visibility:public"],
    deps = if_dpcpp(
        [
            "//itex/core/devices:xpu_device",
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel",
            "//itex/core/profiler:gpu_profiler",
        ],
    ) + [
        "//itex/core:protos_all_cc",
        "//itex/core/kernels:libitex_common",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)

itex_xpu_binary(
    name = "libitex_xla_extension.so",
    linkshared = True,
    set_target = "xla_extension",
    visibility = ["//visibility:public"],
    deps = if_dpcpp(
        [
            "//itex/core/compiler/xla/pjrt:xpu_pjrt_client",
            "@local_config_tf//:jax_internal",
        ],
    ),
)

itex_xpu_binary(
    name = "libitex_gpu_cc.so",
    linkshared = True,
    set_target = "gpu_cc_backend",
    visibility = ["//visibility:public"],
    deps = if_dpcpp(
        [
            "//itex/core/devices:xpu_device",
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel_cc",
            "//itex/core/profiler:gpu_profiler",
        ],
    ) + [
        "//itex/core:protos_all_cc",
        "//itex/core/kernels:itex_common_cc",
    ],
)

cc_binary(
    name = "libitex_cpu.so",
    srcs = [
        "//itex/core/wrapper:itex_cpu_wrapper",
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow"],
    linkshared = True,
    set_target = "dummy",
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:device_backend_util_hdr",
        "//itex/core/kernels:libitex_common",
        "//itex/core/utils:cpu_info",
        "//itex/core/utils:env_var",
        "//itex/core/utils:logging",
    ],
)

cc_binary(
    name = "libitex_cpu_internal_avx512.so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow",
        "-DINTEL_CPU_ONLY",
    ],
    linkshared = True,
    set_target = "cpu_avx512_backend",
    visibility = ["//visibility:public"],
    deps = if_dpcpp(
        [],
        [
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel",
        ],
    ) + [
        "//itex/core/kernels:libitex_common",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)

cc_binary(
    name = "libitex_cpu_internal_avx2.so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow",
        "-DINTEL_CPU_ONLY",
    ],
    linkshared = True,
    set_target = "cpu_avx2_backend",
    visibility = ["//visibility:public"],
    deps = if_dpcpp(
        [],
        [
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel",
        ],
    ) + [
        "//itex/core/kernels:libitex_common",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)

cc_binary(
    name = "libitex_cpu_cc.so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/",
        "-DINTEL_CPU_ONLY",
    ],
    linkshared = True,
    set_target = "cpu_cc_backend",
    visibility = ["//visibility:public"],
    deps = if_dpcpp(
        [],
        [
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel_cc",
        ],
    ) + [
        "//itex/core/kernels:itex_common_cc",
    ],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "xpu_build",
    define_values = {"build_as_xpu": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_for_jax",
    define_values = {"build_for_jax": "true"},
    visibility = ["//visibility:public"],
)

itex_xpu_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
        "//itex/core/utils/lib/random:philox",
        "//itex/core/utils/lib/random:philox_random",
        "//itex/core/utils/onednn:onednn_layout_util",
        "//third_party/eigen3",
        "@local_config_tf//:tf_header_lib",
    ] + if_dpcpp(
        [
            "//itex/core/devices/gpu:eigen_stream_device",
            "//itex/core/devices/gpu:gpu_device_impl",
        ],
    ),
    alwayslink = True,
)
