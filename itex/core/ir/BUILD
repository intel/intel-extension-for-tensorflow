load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(default_visibility = ["//visibility:public"])

td_library(
    name = "InterfacesTdFiles",
    srcs = ["interfaces.td"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "InterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    deps = [
        ":InterfacesTdFiles",
    ],
)

# ODS (https://mlir.llvm.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "dialect.td",
        "ops.td",
    ],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
                "-dialect",
                "tfg",
            ],
            "ops.h.inc",
        ),
        (
            [
                "-gen-op-defs",
                "-dialect",
                "tfg",
            ],
            "ops.cc.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect",
                "tfg",
            ],
            "dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect",
                "tfg",
            ],
            "dialect.cc.inc",
        ),
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect",
                "tfg",
            ],
            "attributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect",
                "tfg",
            ],
            "attributes.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":DialectTdFiles",
        ":InterfacesTdFiles",
        "//itex/core/ir/types:DialectTdFiles",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "interfaces.cc",
        "ops.cc",
        "tf_op_names.cc",
        "tf_op_names.inc",
        "tf_op_wrapper.cc",
        "utility.cc",
    ],
    hdrs = [
        "dialect.h",
        "interfaces.h",
        "ops.h",
        "tf_op_wrapper.h",
        "utility.h",
    ],
    deps = [
        ":DialectIncGen",
        ":InterfacesIncGen",
        "//itex/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_op_registry",
    srcs = ["tf_op_registry.cc"],
    hdrs = ["tf_op_registry.h"],
    deps = [
        ":Dialect",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["utils/shape_inference_utils.cc"],
    hdrs = ["utils/shape_inference_utils.h"],
    deps = [
        ":Dialect",
        "//itex/core/ir/importexport:convert_tensor",
        "//itex/core/ir/importexport:convert_types",
        "//itex/core/ir/importexport:graphdef_export",
        "//itex/core/ir/types:Dialect",
        "//itex/core/utils:common_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@local_config_tf//:protos_all",
    ],
)
