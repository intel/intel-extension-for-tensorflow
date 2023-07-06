load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/utils:common_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@mlir-hlo//:convert_op_folder",
        "@mlir-hlo//:lhlo",
        "@mlir-hlo//:mlir_hlo",
    ],
)

cc_library(
    name = "type_to_shape",
    srcs = ["type_to_shape.cc"],
    hdrs = ["type_to_shape.h"],
    deps = [
        "@mlir-hlo//:mlir_hlo",
        #"//itex/core/compiler/mlir/tensorflow:convert_tensor",
        #"//itex/core/compiler/mlir/tensorflow:convert_type",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mhlo_to_lhlo_with_xla",
    srcs = ["transforms/mhlo_to_lhlo_with_xla.cc"],
    hdrs = ["transforms/mhlo_to_lhlo_with_xla.h"],
    deps = [
        ":attribute_importer",
        ":hlo_module_importer",
        ":hlo_utils",
        ":mlir_hlo_to_hlo",
        ":type_to_shape",
        "//itex/core/compiler/mlir/tensorflow:error_util",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/compiler/xla/service:buffer_assignment",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_parser",
        "//itex/core/compiler/xla/service/gpu:ir_emission_utils",
        "//itex/core/compiler/xla/service/gpu:mkl",
        "//itex/core/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TranslateLib",
        "@mlir-hlo//:lhlo",
        "@mlir-hlo//:lhlo_gpu",
        "@mlir-hlo//:mlir_hlo",
    ],
)

cc_library(
    name = "hlo_module_importer",
    srcs = [
        "hlo_function_importer.cc",
        "hlo_module_importer.cc",
    ],
    hdrs = [
        "hlo_function_importer.h",
        "hlo_module_importer.h",
    ],
    deps = [
        ":attribute_importer",
        ":hlo_utils",
        "//itex/core/compiler/mlir/tensorflow:error_util",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:protobuf_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@mlir-hlo//:mlir_hlo",
    ],
)

cc_library(
    name = "attribute_importer",
    srcs = ["attribute_importer.cc"],
    hdrs = ["attribute_importer.h"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@llvm-project//mlir:IR",
        "@mlir-hlo//:mlir_hlo",
    ],
)

cc_library(
    name = "attribute_exporter",
    srcs = ["attribute_exporter.cc"],
    hdrs = ["attribute_exporter.h"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//protos:tf_protos_all_cc",
        "@llvm-project//mlir:IR",
        "@mlir-hlo//:lhlo_gpu",
        "@mlir-hlo//:mlir_hlo",
    ],
)

cc_binary(
    name = "operator_writer_gen",
    srcs = ["operator_writer_gen.cc"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl_cc_library(
    name = "operator_writer_inc",
    tbl_outs = [([], "operator_writers.inc")],
    tblgen = ":operator_writer_gen",
    td_file = "@mlir-hlo//:mhlo/IR/hlo_ops.td",
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@mlir-hlo//:hlo_ops_td_files",
    ],
)

cc_library(
    name = "mlir_hlo_to_hlo",
    srcs = [
        "mlir_hlo_to_hlo.cc",
        "operator_writers.inc",
    ],
    hdrs = ["mlir_hlo_to_hlo.h"],
    deps = [
        ":attribute_exporter",
        # ":location_exporter",
        ":type_to_shape",
        # ":xla_passes",
        "//itex/core/compiler/mlir:name_utils",
        "@mlir-hlo//:mlir_hlo",
        "@mlir-hlo//:mhlo_passes",
        "//itex/core/compiler/mlir/tensorflow:convert_type",
        "//itex/core/compiler/mlir/tensorflow:error_util",
        "//itex/core/compiler/mlir/xla/utils:xla_helpers",
        "//itex/core/compiler/mlir/xla/utils:layout_util",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla/client/lib:matrix",
        "//itex/core/compiler/xla/client/lib:quantize",
        "//itex/core/compiler/xla/client/lib:slicing",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/compiler/xla/service:hlo_parser",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "//itex/core/utils:common_utils",
    ],
)
