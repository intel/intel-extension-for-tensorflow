package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":timespan",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_visitor",
    srcs = ["xplane_visitor.cc"],
    hdrs = ["xplane_visitor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":timespan",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    deps = [
    ],
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":parse_annotation",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//itex/core/utils:common_utils",
        "//itex/core/utils:logging",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "parse_annotation",
    srcs = ["parse_annotation.cc"],
    hdrs = ["parse_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_schema",
    srcs = ["xplane_schema.cc"],
    hdrs = ["xplane_schema.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":context_types",
        ":tf_op_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "context_types",
    srcs = ["context_types.cc"],
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)
