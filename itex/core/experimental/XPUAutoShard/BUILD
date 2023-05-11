load("//itex:itex.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# ODS (https://mlir.llvm.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "src/xpuautoshard/common/mlir/attributes.td",
        "src/xpuautoshard/common/mlir/dialect.td",
        "src/xpuautoshard/common/mlir/ops.td",
    ],
    includes = ["src"],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "src/xpuautoshard/common/mlir/dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "src/xpuautoshard/common/mlir/dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "src/xpuautoshard/common/mlir/dialect.td",
    deps = [
        ":DialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "AttributesIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "src/xpuautoshard/common/mlir/attributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "src/xpuautoshard/common/mlir/attributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "src/xpuautoshard/common/mlir/attributes.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "OpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "src/xpuautoshard/common/mlir/ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "src/xpuautoshard/common/mlir/ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "src/xpuautoshard/common/mlir/ops.td",
    deps = [":DialectTdFiles"],
)

filegroup(
    name = "xpuautoshard_src",
    srcs = glob(
        [
            "src/xpuautoshard/common/*",
            "src/xpuautoshard/common/*/*",
            "src/xpuautoshard/common/*/*/*",
            "src/xpuautoshard/tensorflow/interface_mlir.*",
            "src/xpuautoshard/tensorflow/passes/*",
        ],
    ),
)

cc_library(
    name = "xpuautoshard",
    srcs = [":xpuautoshard_src"],
    includes = [
        "include",
        "src",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":AttributesIncGen",
        ":DialectIncGen",
        ":OpsIncGen",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/ir:Dialect",
        "//itex/core/utils:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfacesIncGen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)
