load(
    "//itex/core/utils:build_config.bzl",
    "tf_proto_library",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "sycl_platform_id",
    srcs = ["sycl_platform_id.cc"],
    hdrs = ["sycl_platform_id.h"],
    deps = ["//itex/core/compiler/xla/stream_executor:platform"],
)

cc_library(
    name = "sycl_platform",
    srcs = ["sycl_platform.cc"],
    hdrs = ["sycl_platform.h"],
    visibility = ["//visibility:public"],
    deps =
        [
            ":sycl_executor",
            ":sycl_platform_id",
            "//itex/core/compiler/xla/stream_executor",  # buildcleaner: keep
            "//itex/core/compiler/xla/stream_executor:executor_cache",
            "//itex/core/compiler/xla/stream_executor:multi_platform_manager",
            "//itex/core/compiler/xla/stream_executor:stream_executor_pimpl_header",
            "//itex/core/compiler/xla/stream_executor/lib",
            "//itex/core/compiler/xla/stream_executor/platform",
        ] + [
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/synchronization",
        ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "sycl_types_header",
    hdrs = ["sycl_types.h"],
    deps = [
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_executor_header",
    textual_hdrs = ["sycl_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_kernel",
        "//itex/core/compiler/xla/stream_executor:event",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_kernel",
    srcs = ["sycl_kernel.cc"],
    hdrs = ["sycl_kernel.h"],
    deps = [
        ":sycl_types_header",
        "//itex/core/compiler/xla/stream_executor:event",
        "//itex/core/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_event",
    srcs = ["sycl_event.cc"],
    hdrs = ["sycl_event.h"],
    deps = [
        ":sycl_executor_header",
        ":sycl_stream",
        "//itex/core/compiler/xla/stream_executor:stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
    ],
)

cc_library(
    name = "sycl_stream",
    srcs = ["sycl_stream.cc"],
    hdrs = ["sycl_stream.h"],
    deps = [
        ":sycl_executor_header",
        "//itex/core/compiler/xla/stream_executor:stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_timer",
    srcs = ["sycl_timer.cc"],
    hdrs = ["sycl_timer.h"],
    deps = [
        ":sycl_executor_header",
        ":sycl_stream",
        "//itex/core/compiler/xla/stream_executor:stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
    ],
)

cc_library(
    name = "sycl_executor",
    srcs = ["sycl_executor.cc"],
    hdrs = ["sycl_executor.h"],
    deps = [
        ":sycl_event",
        ":sycl_kernel",
        ":sycl_platform_id",
        ":sycl_stream",
        ":sycl_timer",
        "@com_google_absl//absl/strings",
        "//itex/core/devices/gpu:gpu_pool_allocator",
        "//itex/core/compiler/xla/stream_executor:event",
        "//itex/core/compiler/xla/stream_executor:stream_executor_internal",
        "//itex/core/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//itex/core/compiler/xla/stream_executor:timer",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/compiler/xla/stream_executor/platform:dso_loader",
    ] + [
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_platform",
    ],
    alwayslink = 1,
)
