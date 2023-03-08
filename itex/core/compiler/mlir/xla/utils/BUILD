package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xla_helpers",
    srcs = [
        "xla_helpers.cc",
    ],
    hdrs = [
        "xla_helpers.h",
    ],
    deps = [
        ":common",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/client/lib:arithmetic",
        "//itex/core/compiler/xla/client/lib:constants",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xla_argument",
    srcs = [
        "xla_argument.cc",
    ],
    hdrs = [
        "xla_argument.h",
    ],
    deps = [
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/service:hlo",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "layout_util",
    srcs = [
        "layout_util.cc",
    ],
    hdrs = [
        "layout_util.h",
    ],
    deps = [
        ":common",
        ":xla_argument",
        ":xla_helpers",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "common",
    srcs = [
        #"literal_util.cc",
        "shape_util.cc",
        "type_util.cc",
    ],
    hdrs = [
        #"literal_util.h",
        "shape_util.h",
        "type_util.h",
    ],
    deps = [
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)
