package(default_visibility = ["//visibility:public"])

cc_library(
    name = "alias_analysis",
    srcs = ["alias_analysis.cc"],
    hdrs = ["alias_analysis.h"],
    deps = [
        ":ir_array",
        ":llvm_type_conversion_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/service:buffer_assignment",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:logical_buffer",
        # "//tensorflow/core:lib",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "llvm_util",
    srcs = ["llvm_util.cc"],
    hdrs = ["llvm_util.h"],
    deps = [
        ":llvm_type_conversion_util",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service:dump",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_module_config",
        "//itex/core/compiler/xla/service:name_uniquer",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "llvm_type_conversion_util",
    hdrs = ["llvm_type_conversion_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "llvm_command_line_options",
    hdrs = ["llvm_command_line_options.h"],
    deps = [
        # "//tensorflow/core/platform:logging",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ir_array",
    srcs = ["ir_array.cc"],
    hdrs = ["ir_array.h"],
    deps = [
        ":llvm_type_conversion_util",
        ":llvm_util",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "llvm_loop",
    srcs = ["llvm_loop.cc"],
    hdrs = ["llvm_loop.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "loop_emitter",
    srcs = ["loop_emitter.cc"],
    hdrs = ["loop_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_loop",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "fused_ir_emitter",
    srcs = ["fused_ir_emitter.cc"],
    hdrs = ["fused_ir_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        ":loop_emitter",
        ":tuple_ops",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service:elemental_ir_emitter",
        "//itex/core/compiler/xla/service:fusion_node_indexing_evaluation",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "dynamic_update_slice_util",
    srcs = ["dynamic_update_slice_util.cc"],
    hdrs = ["dynamic_update_slice_util.h"],
    deps = [
        ":fused_ir_emitter",
        ":ir_array",
        ":llvm_util",
        ":loop_emitter",
        "//itex/core/compiler/xla/service:buffer_assignment",
        "//itex/core/compiler/xla/service:elemental_ir_emitter",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service/gpu:launch_dimensions",
        "//itex/core/compiler/xla/service/gpu:parallel_loop_emitter",
    ],
)

cc_library(
    name = "sort_util",
    srcs = ["sort_util.cc"],
    hdrs = ["sort_util.h"],
    deps = [
        ":ir_array",
        ":kernel_support_library",
        ":llvm_loop",
        ":llvm_util",
        ":loop_emitter",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service/gpu:launch_dimensions",
        "//itex/core/compiler/xla/service/gpu:parallel_loop_emitter",
        "//itex/core/compiler/xla/service/gpu:target_util",
        # "//tensorflow/core:lib",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tuple_ops",
    srcs = ["tuple_ops.cc"],
    hdrs = ["tuple_ops.h"],
    deps = [
        ":ir_array",
        ":llvm_type_conversion_util",
        ":llvm_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "kernel_support_library",
    srcs = ["kernel_support_library.cc"],
    hdrs = ["kernel_support_library.h"],
    deps = [
        ":llvm_loop",
        ":llvm_type_conversion_util",
        ":llvm_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "buffer_assignment_util",
    srcs = ["buffer_assignment_util.cc"],
    hdrs = ["buffer_assignment_util.h"],
    deps = [
        "//itex/core/compiler/xla/service:buffer_assignment",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "math_ops",
    srcs = ["math_ops.cc"],
    hdrs = ["math_ops.h"],
    deps = [
        ":llvm_util",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "ir_builder_mixin",
    srcs = [],
    hdrs = ["ir_builder_mixin.h"],
    deps = [
        "@llvm-project//llvm:Core",
    ],
)
