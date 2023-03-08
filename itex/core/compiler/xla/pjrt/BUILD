load("//itex:itex.bzl", "itex_xpu_binary", "itex_xpu_library", "tf_copts")

cc_library(
    name = "c_api_conversions",
    srcs = ["c_api_conversions.cc"],
    hdrs = ["c_api_conversions.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/compiler/c:pjrt_c_api",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "xpu_pjrt_client",
    srcs = ["xpu_pjrt_client.cc"],
    hdrs = [
        "xpu_pjrt_client.h",
        "//itex/core:itex_version_generator",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_conversions",
        ":pjrt_c_api_helpers",
        ":pjrt_compiler",
        ":pjrt_stream_executor_client",
        ":se_xpu_client",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/service:compiler",
        "//itex/core/compiler/xla/service:computation_placer",
        "//itex/core/compiler/xla/service/gpu:gpu_transfer_manager",
        "//itex/core/compiler/xla/service/gpu:spir_compiler",
        "//itex/core/compiler/xla/stream_executor:sycl_platform",
        "//itex/core/utils:common_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/compiler/c:pjrt_c_api",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla/pjrt:pjrt_client",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "se_xpu_client",
    srcs = ["se_xpu_pjrt_client.cc"],
    hdrs = ["se_xpu_pjrt_client.h"],
    deps = [
        ":xpu_helpers",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:client_library",
        "//itex/core/compiler/xla/pjrt:pjrt_stream_executor_client",
        "//itex/core/compiler/xla/service:platform_util",
        "//itex/core/compiler/xla/service/gpu:gpu_executable_run_options",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "xpu_helpers",
    srcs = ["xpu_helpers.cc"],
    hdrs = ["xpu_helpers.h"],
    deps = [
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:client_library",
        "//itex/core/compiler/xla/client:local_client",
        "//itex/core/compiler/xla/service:platform_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    deps = [
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tracked_device_buffer",
    srcs = ["tracked_device_buffer.cc"],
    hdrs = ["tracked_device_buffer.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":event_pool",
        ":local_device_state",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/service:shaped_buffer",
        "//itex/core/compiler/xla/service:transfer_manager",
        "//itex/core/compiler/xla/stream_executor:device_memory",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/compiler/xla/stream_executor:event",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    deps = [
        ":event_pool",
        ":semaphore",
        ":worker_thread",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:local_client",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:event",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "pjrt_client",
    srcs = ["pjrt_client.cc"],
    hdrs = ["pjrt_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_pool",
        ":pjrt_executable",
        ":pjrt_future",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:executable_build_options",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_cost_analysis",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "pjrt_executable",
    srcs = ["pjrt_executable.cc"],
    hdrs = ["pjrt_executable.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:executable_build_options",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pjrt_compiler",
    srcs = ["pjrt_compiler.cc"],
    hdrs = ["pjrt_compiler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_client",
        ":pjrt_executable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:executable_build_options",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/service:computation_placer",
        "//itex/core/compiler/xla/service:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "pjrt_stream_executor_client",
    srcs = ["pjrt_stream_executor_client.cc"],
    hdrs = ["pjrt_stream_executor_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_pool",
        ":local_device_state",
        #":metrics",
        ":mlir_to_hlo",
        ":pjrt_client",
        ":pjrt_future",
        ":tracked_device_buffer",
        ":transpose",
        ":utils",
        "//itex/core/compiler/xla:cpu_function_runtime",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:executable_build_options",
        "//itex/core/compiler/xla/client:local_client",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/service:computation_layout",
        "//itex/core/compiler/xla/service:computation_placer",
        "//itex/core/compiler/xla/service:executable",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_cost_analysis",
        "//itex/core/compiler/xla/service:maybe_owning_device_memory",
        "//itex/core/compiler/xla/service:shaped_buffer",
        "//itex/core/compiler/xla/service:transfer_manager",
        "//itex/core/compiler/xla/service/gpu:gpu_executable_run_options",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:event",
        "//itex/core/compiler/xla/stream_executor/lib",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pjrt_future",
    hdrs = ["pjrt_future.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "mlir_to_hlo",
    srcs = ["mlir_to_hlo.cc"],
    hdrs = ["mlir_to_hlo.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/compiler/mlir/tensorflow:error_util",
        "//itex/core/compiler/mlir/xla:mlir_hlo_to_hlo",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla/client:xla_computation",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Transforms",
        "@mlir-hlo//:mhlo_passes",
        "@mlir-hlo//:mlir_hlo",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:node_hash_map",
    ],
)

cc_library(
    name = "transpose",
    srcs = [
        "transpose.cc",
        "transpose_kernels.h",
    ],
    hdrs = ["transpose.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lru_cache",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)
