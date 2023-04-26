package(default_visibility = ["//visibility:public"])

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

gentbl_cc_library(
    name = "gpu_transforms_pass_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LMHLOGPUTransforms",
            ],
            "include/transforms/itex_gpu_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/itex_gpu_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "itex_gpu_passes",
    srcs = [
        "include/transforms/itex_gpu_passes.h.inc",
        "transforms/gpu_fusion_rewrite.cc",
        "transforms/gpu_kernel_lowering_passes.cc",
        "transforms/replace_alloc_with_arg.cc",
    ],
    hdrs = [
        "transforms/itex_gpu_passes.h",
    ],
    deps = [
        ":gpu_transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVTransforms",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:Transforms",
        # "@mlir-extensions//:mlir_lib",
        "@mlir-hlo//:mlir_hlo",
        "@mlir-hlo//:transforms_gpu_passes",
        "@mlir-hlo//:transforms_passes",
    ],
)
