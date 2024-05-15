package(
    licenses = ["notice"],  # Apache 2.0
)

load("//itex:itex.bzl", "cc_library", "if_using_nextpluggable_device")

cc_library(
    name = "eigen_runtime_impl",
    srcs = ["eigen_itex_gpu_runtime.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:logging",
        "//third_party/build_option/sycl:eigen_itex_gpu_header",
        "@com_google_absl//absl/synchronization",
        "@itex_eigen_archive//:eigen",
    ],
    alwayslink = True,
)

cc_library(
    name = "itex_gpu_runtime_imp",
    srcs = ["itex_gpu_runtime.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":eigen_runtime_impl",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/utils:logging",
        "//third_party/build_option/sycl:itex_gpu_header",
        "@com_google_absl//absl/synchronization",
        "@itex_eigen_archive//:eigen",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_device_impl",
    srcs = ["gpu_device_plugin.cc"],
    hdrs = [
        "gpu_device_plugin.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_pool_allocator",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_pool_allocator",
    hdrs = [
        "gpu_pool_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":itex_gpu_runtime_imp",
        "//itex/core/devices:bfc_allocator",
        "//itex/core/utils:logging",
        "//third_party/build_option/sycl:itex_gpu_header",
    ],
)

cc_library(
    name = "gpu_info",
    srcs = ["gpu_info.cc"],
    hdrs = ["gpu_info.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:logging",
        "//third_party/build_option/sycl:itex_gpu_header",
        "//third_party/eigen3",
        "@itex_eigen_archive//:eigen",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "eigen_stream_device",
    srcs = ["eigen_stream_device.cc"],
    hdrs = [
        "eigen_stream_device.h",
        "gpu_device_plugin.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_info",
        "//itex/core/utils:logging",
        "//itex/core/utils/gtl:gtl_libs",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@itex_eigen_archive//:eigen",
        "@local_config_tf//:tf_header_lib",
    ] + if_using_nextpluggable_device(
        ["//third_party/build_option/sycl:itex_gpu_header"],
    ),
    alwayslink = True,
)

exports_files(
    srcs = [
        "gpu_device_plugin.h",
    ],
    visibility = ["//visibility:public"],
)
