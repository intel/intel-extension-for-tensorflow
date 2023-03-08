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

# The stream_executor_headers target does not prescribe an implementation.
cc_library(
    name = "stream_executor_headers",
    textual_hdrs = [
        "blas.h",
        "device_description.h",
        "device_memory.h",
        "device_memory_allocator.h",
        "device_options.h",
        "event.h",
        "executor_cache.h",
        "gpu_launch_dim.h",
        "kernel.h",
        "kernel_cache_config.h",
        "kernel_spec.h",
        "launch_dim.h",
        "module_spec.h",
        "multi_platform_manager.h",
        "platform.h",
        "plugin.h",
        "stream.h",
        "stream_executor.h",
        "stream_executor_internal.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
        "timer.h",
        "trace_listener.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator_stats",
        ":host_or_device_scalar",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "launch_dim",
    hdrs = [
        "gpu_launch_dim.h",
        "launch_dim.h",
    ],
    deps = [
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_listener",
    hdrs = [
        "trace_listener.h",
    ],
    deps = [
        ":device_memory",
        ":kernel",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":launch_dim",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "kernel_cache_config",
    hdrs = ["kernel_cache_config.h"],
)

cc_library(
    name = "module_spec",
    hdrs = ["module_spec.h"],
    deps = [
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "allocator_stats",
    srcs = [
        "allocator_stats.cc",
    ],
    hdrs = ["allocator_stats.h"],
    deps = [
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = ["//itex/core/compiler/xla/stream_executor/platform"],
)

cc_library(
    name = "device_options",
    hdrs = ["device_options.h"],
    deps = [
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":plugin",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "plugin",
    srcs = ["plugin.cc"],
    hdrs = ["plugin.h"],
)

cc_library(
    name = "executor_cache",
    srcs = [
        "device_description.h",
        "device_memory.h",
        "device_options.h",
        "event.h",
        "executor_cache.cc",
        "launch_dim.h",
        "plugin.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
    ],
    hdrs = [
        "blas.h",
        "executor_cache.h",
        "kernel.h",
        "kernel_cache_config.h",
        "kernel_spec.h",
        "platform.h",
        "stream.h",
        "stream_executor_internal.h",
        "trace_listener.h",
    ],
    deps = [
        ":allocator_stats",
        ":data_type",
        ":device_description",
        ":device_memory",
        ":device_options",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":plugin",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "multi_platform_manager",
    srcs = ["multi_platform_manager.cc"],
    hdrs = ["multi_platform_manager.h"],
    deps = [
        ":platform",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "stream_executor_internal",
    srcs = [
        "stream_executor_internal.cc",
    ],
    hdrs = [
        "stream_executor_internal.h",
    ],
    deps = [
        ":allocator_stats",
        ":device_description",
        ":device_memory",
        ":device_options",
        ":kernel",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "stream_executor_pimpl_header",
    hdrs = [
        "device_description.h",
        "kernel.h",
        "kernel_cache_config.h",
        "stream_executor_pimpl.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":device_description",
        ":kernel_cache_config",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_internal",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

# It implements :stream_executor_pimpl_header
cc_library(
    name = "stream_executor_pimpl",
    srcs = [
        "stream.cc",
        "stream_executor_pimpl.cc",
    ],
    hdrs = ["stream_executor_pimpl.h"],
    deps = [
        ":blas",
        ":device_memory",
        ":event",
        ":executor_cache",
        ":host_or_device_scalar",
        ":kernel",
        ":launch_dim",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_internal",
        ":temporary_device_memory",
        ":temporary_memory_manager",
        ":timer",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "kernel",
    srcs = [
        "kernel.cc",
        "plugin.h",
        "stream.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
    ],
    hdrs = [
        "blas.h",
        "device_description.h",
        "device_options.h",
        "event.h",
        "kernel.h",
        "kernel_spec.h",
        "launch_dim.h",
        "multi_platform_manager.h",
        "platform.h",
        "stream_executor.h",
        "stream_executor_internal.h",
        "timer.h",
        "trace_listener.h",
    ],
    deps = [
        ":allocator_stats",
        ":data_type",
        ":device_description",
        ":device_memory",
        ":device_options",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":platform",
        ":plugin",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "timer",
    srcs = [
        "device_description.h",
        "kernel_cache_config.h",
        "timer.cc",
    ],
    hdrs = [
        "blas.h",
        "kernel.h",
        "stream.h",
        "stream_executor.h",
        "timer.h",
    ],
    deps = [
        ":data_type",
        ":device_description",
        ":kernel_cache_config",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_pimpl_header",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "blas",
    srcs = ["blas.cc"],
    hdrs = ["blas.h"],
    deps = [
        ":data_type",
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "host_or_device_scalar",
    hdrs = ["host_or_device_scalar.h"],
    deps = [
        ":data_type",
        ":device_memory",
        "//itex/core/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "event",
    srcs = [
        "blas.h",
        "device_description.h",
        "device_options.h",
        "event.cc",
        "kernel_cache_config.h",
        "launch_dim.h",
        "plugin.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
        "trace_listener.h",
    ],
    hdrs = [
        "device_memory.h",
        "event.h",
        "kernel.h",
        "kernel_spec.h",
        "platform.h",
        "stream.h",
        "stream_executor_internal.h",
    ],
    deps = [
        ":allocator_stats",
        ":blas",
        ":data_type",
        ":device_description",
        ":device_memory",
        ":device_options",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":platform",
        ":plugin",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "temporary_memory_manager",
    srcs = ["temporary_memory_manager.cc"],
    hdrs = ["temporary_memory_manager.h"],
    deps = [
        ":device_memory",
        ":stream_executor_headers",
        ":stream_executor_pimpl_header",
        ":temporary_device_memory",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "temporary_device_memory",
    srcs = [
        "event.h",
        "temporary_device_memory.cc",
        "temporary_memory_manager.h",
    ],
    hdrs = ["temporary_device_memory.h"],
    deps = [
        ":device_memory",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "stream_executor",
    textual_hdrs = [
        "blas.h",
        "device_description.h",
        "device_memory.h",
        "device_memory_allocator.h",
        "device_options.h",
        "event.h",
        "executor_cache.h",
        "gpu_launch_dim.h",
        "kernel.h",
        "kernel_cache_config.h",
        "kernel_spec.h",
        "launch_dim.h",
        "module_spec.h",
        "multi_platform_manager.h",
        "platform.h",
        "plugin.h",
        "stream.h",
        "stream_executor.h",
        "stream_executor_internal.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
        "timer.h",
        "trace_listener.h",
    ],
    deps = [":stream_executor_headers"] + [":stream_executor_impl"],
)

cc_library(
    name = "stream_executor_impl",
    deps = [
        ":device_description",
        ":device_memory",
        ":event",
        ":kernel",
        ":launch_dim",
        ":multi_platform_manager",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_pimpl",
        ":timer",
    ],
)

cc_library(
    name = "device_memory_allocator",
    hdrs = ["device_memory_allocator.h"],
    deps = [
        ":device_memory",
        ":platform",
        ":stream_executor",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scratch_allocator",
    srcs = ["scratch_allocator.cc"],
    hdrs = ["scratch_allocator.h"],
    deps = [
        ":device_memory",
        ":stream_executor_headers",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

exports_files(["lazy_op_runner.h"])

alias(
    name = "sycl_platform",
    actual = "//itex/core/compiler/xla/stream_executor/sycl:all_runtime",
)

alias(
    name = "host_platform",
    actual = "//itex/core/compiler/xla/stream_executor/host:all_runtime",
)
