package(default_visibility = ["//visibility:public"])

cc_library(
    name = "spmd_partitioner",
    srcs = [
        "convolution_handler.cc",
        "custom_call_handler.cc",
        "dot_handler.cc",
        "fft_handler.cc",
        "gather_scatter_handler.cc",
        "spmd_partitioner.cc",
        "spmd_partitioner_util.cc",
    ],
    hdrs = [
        "convolution_handler.h",
        "custom_call_handler.h",
        "spmd_partitioner.h",
        "spmd_partitioner_util.h",
    ],
    deps = [
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:protobuf_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client/lib:comparators",
        "//itex/core/compiler/xla/service:dot_as_convolution_util",
        "//itex/core/compiler/xla/service:flatten_call_graph",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_cse",
        "//itex/core/compiler/xla/service:hlo_dce",
        "//itex/core/compiler/xla/service:hlo_lexer",
        "//itex/core/compiler/xla/service:hlo_pass",
        "//itex/core/compiler/xla/service:hlo_pass_pipeline",
        "//itex/core/compiler/xla/service:hlo_query",
        "//itex/core/compiler/xla/service:hlo_sharding_util",
        "//itex/core/compiler/xla/service:pattern_matcher",
        "//itex/core/compiler/xla/service:shape_inference",
        "//itex/core/compiler/xla/service:sharding_propagation",
        "//itex/core/compiler/xla/service:tuple_simplifier",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "canonicalize_all_gather_for_cse",
    srcs = ["canonicalize_all_gather_for_cse.cc"],
    hdrs = ["canonicalize_all_gather_for_cse.h"],
    deps = [
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_pass",
        "//itex/core/compiler/xla/service:hlo_query",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "schedule_aware_collective_ops_cse",
    srcs = ["schedule_aware_collective_ops_cse.cc"],
    hdrs = ["schedule_aware_collective_ops_cse.h"],
    deps = [
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "stateful_rng_spmd_partitioner",
    srcs = ["stateful_rng_spmd_partitioner.cc"],
    hdrs = ["stateful_rng_spmd_partitioner.h"],
    deps = [
        ":spmd_partitioner",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/memory",
    ],
)
