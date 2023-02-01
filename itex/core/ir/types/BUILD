load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(default_visibility = ["//visibility:public"])

# ODS (https://mlir.llvm.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "attributes.td",
        "dialect.td",
        "types.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:SubElementInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [
        ":DialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "AttributesIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "attributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "attributes.cc.inc",
        ),
        (
            ["-gen-enum-decls"],
            "attributes_enum.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "attributes_enum.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attributes.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "TypesIncGen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "types.td",
    deps = [":DialectTdFiles"],
)

cc_library(
    name = "Dialect",
    srcs = ["dialect.cc"],
    hdrs = [
        "dialect.h",
    ],
    textual_hdrs = [
        "types.def",
    ],
    deps = [
        ":AttributesIncGen",
        ":DialectIncGen",
        ":TypesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
