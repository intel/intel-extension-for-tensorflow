package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bit_cast",
    hdrs = ["bit_cast.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "comparison_util",
    srcs = [
        "comparison_util.cc",
    ],
    hdrs = [
        "comparison_util.h",
        "primitive_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "execution_options_util",
    srcs = [
        "execution_options_util.cc",
    ],
    hdrs = [
        "execution_options_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":debug_options_flags",
        "//protos:tf_protos_all_cc",
        "//protos:xla_protos_all_cc",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "service_interface",
    srcs = [],
    hdrs = ["service_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        "//protos:tf_protos_all_cc",
        "//protos:xla_protos_all_cc",
    ],
)

cc_library(
    name = "status_macros",
    srcs = ["status_macros.cc"],
    hdrs = ["status_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":statusor",
        ":types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "statusor",
    hdrs = [
        "statusor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = [
        "iterator_util.h",
        "map_util.h",
        "overflow_util.h",
        "util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        ":status_macros",
        ":statusor",
        ":types",
        "//itex/core/utils:common_utils",
        "//protos:xla_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "permutation_util",
    srcs = ["permutation_util.cc"],
    hdrs = ["permutation_util.h"],
    deps = [
        ":types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "protobuf_util",
    srcs = ["protobuf_util.cc"],
    hdrs = [
        "protobuf_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "shape_util",
    srcs = [
        "index_util.cc",
        "layout.cc",
        "layout_util.cc",
        "primitive_util.cc",
        "shape.cc",
        "shape_util.cc",
    ],
    hdrs = [
        "index_util.h",
        "layout.h",
        "layout_util.h",
        "primitive_util.h",
        "shape.h",
        "shape_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":permutation_util",
        ":protobuf_util",
        ":status",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "//protos:xla_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sharding_op_util",
    srcs = ["sharding_op_util.cc"],
    hdrs = ["sharding_op_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status_macros",
        ":util",
        "//itex/core/compiler/xla/service:hlo_lexer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "literal",
    srcs = ["literal.cc"],
    hdrs = ["literal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":bit_cast",
        ":permutation_util",
        ":shape_util",
        ":status_macros",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "literal_util",
    srcs = ["literal_util.cc"],
    hdrs = ["literal_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal",
        ":shape_util",
        ":status_macros",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "error_spec",
    hdrs = ["error_spec.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "literal_comparison",
    srcs = ["literal_comparison.cc"],
    hdrs = ["literal_comparison.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":error_spec",
        ":literal",
        ":literal_util",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "metric_table_report",
    srcs = ["metric_table_report.cc"],
    hdrs = ["metric_table_report.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "device_util",
    hdrs = ["device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "array2d",
    hdrs = ["array2d.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "array3d",
    hdrs = ["array3d.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":types",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "array4d",
    hdrs = ["array4d.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":array2d",
        ":types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable_run_options",
    srcs = ["executable_run_options.cc"],
    hdrs = ["executable_run_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "packed_literal_reader",
    srcs = ["packed_literal_reader.cc"],
    hdrs = ["packed_literal_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "text_literal_reader",
    srcs = ["text_literal_reader.cc"],
    hdrs = ["text_literal_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        "//itex/core/compiler/xla/service:hlo_parser",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "text_literal_writer",
    srcs = ["text_literal_writer.cc"],
    hdrs = ["text_literal_writer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "shape_tree",
    srcs = ["shape_tree.cc"],
    hdrs = ["shape_tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        ":status_macros",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "shape_layout",
    srcs = ["shape_layout.cc"],
    hdrs = ["shape_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        ":types",
        ":util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "window_util",
    srcs = ["window_util.cc"],
    hdrs = ["window_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//itex/core/utils:common_utils",
        "//protos:xla_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "reference_util",
    srcs = ["reference_util.cc"],
    hdrs = ["reference_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal_util",
        ":util",
        ":window_util",
        "//itex/core/compiler/xla/client:padding",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_evaluator",
        "//itex/core/compiler/xla/service:shape_inference",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "parse_flags_from_env",
    srcs = ["parse_flags_from_env.cc"],
    hdrs = ["parse_flags_from_env.h"],
    deps =
        [
            ":types",
            "//itex/core/utils:common_utils",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/types:span",
        ],
)

cc_library(
    name = "debug_options_flags",
    srcs = [
        "debug_options_flags.cc",
        "debug_options_parsers.h",
    ],
    hdrs = ["debug_options_flags.h"],
    visibility = ["//visibility:public"],
    deps =
        [
            ":parse_flags_from_env",
            "//itex/core/utils:common_utils",
            "//protos:xla_protos_all_cc",
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:node_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
        ],
)

cc_library(
    name = "cpu_function_runtime",
    srcs = ["cpu_function_runtime.cc"],
    hdrs = ["cpu_function_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "refcounting_hash_map",
    hdrs = ["refcounting_hash_map.h"],
    deps = [
        ":statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
)

cc_library(
    name = "side_effect_util",
    srcs = ["side_effect_util.cc"],
    hdrs = ["side_effect_util.h"],
)

cc_library(
    name = "test",
    testonly = 1,
    hdrs = ["test.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "//itex/core/utils:test",
    ],
)

cc_library(
    name = "test_helpers",
    testonly = 1,
    hdrs = ["test_helpers.h"],
    deps = [
        ":statusor",
        ":types",
        "//itex/core/utils:common_utils",
        "//itex/core/utils:test",
        "@com_google_absl//absl/strings",
    ],
)
