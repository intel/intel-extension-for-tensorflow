load("//itex:itex.bzl", "cc_library")
load("@itex-llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

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
        "@itex-llvm-project//mlir:CallInterfacesTdFiles",
        "@itex-llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@itex-llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@itex-llvm-project//mlir:OpBaseTdFiles",
        "@itex-llvm-project//mlir:SideEffectInterfacesTdFiles",
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
    tblgen = "@itex-llvm-project//mlir:mlir-tblgen",
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
    tblgen = "@itex-llvm-project//mlir:mlir-tblgen",
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
    tblgen = "@itex-llvm-project//mlir:mlir-tblgen",
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
        "@itex-llvm-project//llvm:Support",
        "@itex-llvm-project//mlir:ControlFlowInterfacesIncGen",
        "@itex-llvm-project//mlir:Dialect",
        "@itex-llvm-project//mlir:IR",
        "@itex-llvm-project//mlir:Pass",
        "@itex-llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)
