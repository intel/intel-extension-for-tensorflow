load("//itex:itex.bzl", "itex_xpu_binary", "itex_xpu_library", "tf_copts")

itex_xpu_binary(
    name = "libitex_gpu.so",
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//itex/core/devices:xpu_device",
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel",
            "//itex/core/profiler:gpu_profiler",
        ],
        "//conditions:default": [],
    }) + [
        "@local_config_tf//:_pywrap_tensorflow_internal",
        "//itex/core:protos_all_cc",
        "//itex/core/kernels:libitex_common",
    ],
)

itex_xpu_binary(
    name = "libitex_xla_extension.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            # "//itex/core/compiler/xla/pjrt:xpu_pjrt_client",
            "@local_config_tf//:jax_internal",
        ],
        "//conditions:default": [],
    }),
)

itex_xpu_binary(
    name = "itex_gpu_cc",
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//itex/core/devices:xpu_device",
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel_cc",
            "//itex/core/profiler:gpu_profiler",
        ],
        "//conditions:default": [],
    }) + [
        "//itex/core:protos_all_cc",
        "//itex/core/kernels:itex_common_cc",
    ],
)

cc_binary(
    name = "libitex_cpu.so",
    copts = [
        "-DINTEL_CPU_ONLY",
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [],
        "//conditions:default": [
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel",
        ],
    }) + [
        "@local_config_tf//:_pywrap_tensorflow_internal",
        "//itex/core/kernels:libitex_common",
    ],
)

cc_binary(
    name = "itex_cpu_cc",
    copts = [
        "-DINTEL_CPU_ONLY",
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_tensorflow"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [],
        "//conditions:default": [
            "//itex/core/graph:xpu_graph",
            "//itex/core/kernels:xpu_kernel_cc",
        ],
    }) + [
        "//itex/core/kernels:itex_common_cc",
    ],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
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
        "//itex/core/utils/lib/random:philox",
        "//itex/core/utils/lib/random:philox_random",
        "//itex/core/utils:common_utils",
        "//itex/core/utils/onednn:onednn_layout_util",
        "//third_party/eigen3",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//itex/core/devices/gpu:eigen_stream_device",
            "//itex/core/devices/gpu:gpu_device_impl",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)
