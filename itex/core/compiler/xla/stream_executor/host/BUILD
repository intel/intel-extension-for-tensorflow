# Description:
#   Host-platform specific StreamExecutor support code.

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "host_platform_id",
    srcs = [
        "host_platform_id.cc",
    ],
    hdrs = [
        "host_platform_id.h",
    ],
    deps = [
        "//itex/core/compiler/xla/stream_executor:platform",
    ],
)

cc_library(
    name = "host_platform",
    srcs = [
        "host_platform.cc",
    ],
    hdrs = [
        "host_platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":host_gpu_executor",
        ":host_platform_id",
        "//itex/core/compiler/xla/stream_executor:executor_cache",
        "//itex/core/compiler/xla/stream_executor:multi_platform_manager",
        "//itex/core/compiler/xla/stream_executor:stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "host_stream",
    srcs = [
        "host_stream.cc",
    ],
    hdrs = [
        "host_stream.h",
    ],
    deps = [
        "//itex/core/compiler/xla/stream_executor:kernel",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "host_timer",
    srcs = [
        "host_timer.cc",
    ],
    hdrs = [
        "host_timer.h",
    ],
    deps = [
        "//itex/core/compiler/xla/stream_executor:stream_executor_internal",
        "//itex/core/compiler/xla/stream_executor:timer",
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

# TODO(22689637): Rename this target.
cc_library(
    name = "host_gpu_executor",
    srcs = [
        "host_gpu_executor.cc",
    ],
    hdrs = [
        "host_gpu_executor.h",
    ],
    deps = [
        ":host_platform_id",
        ":host_stream",
        ":host_timer",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:kernel",
        "//itex/core/compiler/xla/stream_executor:rng",
        "//itex/core/compiler/xla/stream_executor:stream_executor_internal",
        "//itex/core/compiler/xla/stream_executor:stream_executor_pimpl",
        "//itex/core/compiler/xla/stream_executor:timer",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)
