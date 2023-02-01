package(default_visibility = ["//visibility:public"])

cc_library(
    name = "import",
    srcs = [
        "functiondef_import.cc",
        "import.cc",
    ],
    hdrs = [
        "functiondef_import.h",
        "import.h",
    ],
    deps = [
        ":convert_attributes",
        ":convert_tensor",
        ":convert_types",
        "//itex/core/utils:common_utils",
        "@local_config_tf//:protos_all",
        # "//tensorflow/core/common_runtime:function_def_utils",
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "export",
    srcs = [
        "export.cc",
        "functiondef_export.cc",
    ],
    hdrs = [
        "export.h",
        "functiondef_export.h",
    ],
    deps = [
        ":convert_attributes",
        ":convert_tensor",
        ":convert_types",
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "parse_text_proto",
    srcs = [
        "parse_text_proto.cc",
    ],
    hdrs = ["parse_text_proto.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mangling",
    srcs = [
        "mangling.cc",
    ],
    hdrs = ["mangling.h"],
    deps = [
        ":parse_text_proto",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "convert_types",
    srcs = [
        "convert_types.cc",
    ],
    hdrs = ["convert_types.h"],
    deps = [
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = [
        "convert_tensor.cc",
    ],
    hdrs = ["convert_tensor.h"],
    deps = [
        ":convert_types",
        ":mangling",
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "graphdef_import",
    srcs = ["graphdef_import.cc"],
    hdrs = ["graphdef_import.h"],
    deps = [
        ":convert_attributes",
        ":convert_types",
        ":import",
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "graphdef_export",
    srcs = ["graphdef_export.cc"],
    hdrs = ["graphdef_export.h"],
    deps = [
        ":convert_attributes",
        ":convert_types",
        ":export",
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "convert_attributes",
    srcs = [
        "convert_attributes.cc",
    ],
    hdrs = ["convert_attributes.h"],
    deps = [
        ":convert_tensor",
        ":convert_types",
        ":mangling",
        "//itex/core/ir:Dialect",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@local_config_tf//:protos_all",
    ],
)
