load("//itex:itex.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tfg_passes_builder",
    srcs = ["tfg_passes_builder.cc"],
    hdrs = ["tfg_passes_builder.h"],
    deps = [
        "@itex-llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "tfg_optimizer_hook",
    srcs = ["tfg_optimizer_hook.cc"],
    hdrs = ["tfg_optimizer_hook.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tfg_passes_builder",
        "//itex/core/experimental/XPUAutoShard:xpuautoshard",
        "//itex/core/graph:optimizer_config",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:node_type_attr_map",
        "//itex/core/graph/utils:symbolic_shapes",
        "//itex/core/ir:Dialect",
        "//itex/core/ir:tf_op_registry",
        "//itex/core/ir/importexport:graphdef_export",
        "//itex/core/ir/importexport:graphdef_import",
        "//itex/core/utils:common_utils",
        "@itex-llvm-project//mlir:Pass",
        "@itex-llvm-project//mlir:Transforms",
    ],
)
