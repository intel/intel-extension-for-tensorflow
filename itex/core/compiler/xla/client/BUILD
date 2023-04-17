package(default_visibility = ["//visibility:public"])

cc_library(
    name = "global_data",
    srcs = ["global_data.cc"],
    hdrs = ["global_data.h"],
    deps = [
        "//itex/core/compiler/xla:service_interface",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        ":global_data",
        ":xla_computation",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:execution_options_util",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:service_interface",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable_build_options",
    srcs = ["executable_build_options.cc"],
    hdrs = ["executable_build_options.h"],
    deps = [
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:execution_options_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service:computation_placer",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "local_client",
    srcs = ["local_client.cc"],
    hdrs = ["local_client.h"],
    deps = [
        ":client",
        ":executable_build_options",
        ":xla_computation",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/service:backend",
        "//itex/core/compiler/xla/service:compiler",
        "//itex/core/compiler/xla/service:dump",
        "//itex/core/compiler/xla/service:executable",
        "//itex/core/compiler/xla/service:local_service",
        "//itex/core/compiler/xla/service:maybe_owning_device_memory",
        "//itex/core/compiler/xla/service:shaped_buffer",
        "//itex/core/compiler/xla/service:source_map_util",
        "//itex/core/compiler/xla/service:stream_pool",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "compile_only_client",
    srcs = ["compile_only_client.cc"],
    hdrs = ["compile_only_client.h"],
    deps = [
        ":client",
        ":xla_computation",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/service:compile_only_service",
        "//itex/core/compiler/xla/service:compiler",
        "//itex/core/compiler/xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

# This target is used to instantiate the XLA service in-process and create
# a client for it.
cc_library(
    name = "client_library",
    srcs = ["client_library.cc"],
    hdrs = ["client_library.h"],
    deps = [
        ":compile_only_client",
        ":local_client",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service:backend",
        "//itex/core/compiler/xla/service:compile_only_service",
        "//itex/core/compiler/xla/service:local_service",
        "//itex/core/compiler/xla/service:platform_util",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "sharding_builder",
    srcs = ["sharding_builder.cc"],
    hdrs = ["sharding_builder.h"],
    deps = [
        "//itex/core/compiler/xla:array",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//protos:tf_protos_all_cc",
    ],
)

cc_library(
    name = "xla_computation",
    srcs = ["xla_computation.cc"],
    hdrs = ["xla_computation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "xla_builder",
    srcs = ["xla_builder.cc"],
    hdrs = ["xla_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":padding",
        ":sharding_builder",
        ":xla_computation",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:execution_options_util",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:sharding_op_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:shape_inference",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
