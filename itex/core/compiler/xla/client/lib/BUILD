package(default_visibility = ["//visibility:public"])

cc_library(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        ":constants",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client:xla_computation",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "comparators",
    srcs = ["comparators.cc"],
    hdrs = [
        "comparators.h",
    ],
    deps = [
        ":constants",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client:xla_computation",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "constants",
    srcs = ["constants.cc"],
    hdrs = ["constants.h"],
    deps = [
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
    ],
)

cc_library(
    name = "broadcast",
    srcs = ["broadcast.cc"],
    hdrs = ["broadcast.h"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_grad_size_util",
    srcs = ["conv_grad_size_util.cc"],
    hdrs = ["conv_grad_size_util.h"],
    deps = [
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla/client:padding",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "dynamic_shaped_ops",
    srcs = ["dynamic_shaped_ops.cc"],
    hdrs = ["dynamic_shaped_ops.h"],
    deps = [
        ":constants",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:value_inference",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "loops",
    srcs = ["loops.cc"],
    hdrs = ["loops.h"],
    deps = [
        ":constants",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client:xla_computation",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "math",
    srcs = ["math.cc"],
    hdrs = ["math.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":loops",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla/client:xla_builder",
    ],
)

cc_library(
    name = "matrix",
    srcs = ["matrix.cc"],
    hdrs = ["matrix.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":slicing",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":conv_grad_size_util",
        "//itex/core/compiler/xla/client:xla_builder",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "prng",
    srcs = ["prng.cc"],
    hdrs = ["prng.h"],
    deps = [
        ":constants",
        ":math",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "qr",
    srcs = ["qr.cc"],
    hdrs = ["qr.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":slicing",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "lu_decomposition",
    srcs = ["lu_decomposition.cc"],
    hdrs = ["lu_decomposition.h"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
    ],
)

cc_library(
    name = "approx_topk",
    srcs = ["approx_topk.cc"],
    hdrs = ["approx_topk.h"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client:xla_computation",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "slicing",
    srcs = ["slicing.cc"],
    hdrs = ["slicing.h"],
    deps = [
        ":arithmetic",
        ":constants",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sorting",
    srcs = ["sorting.cc"],
    hdrs = ["sorting.h"],
    deps = [
        ":comparators",
        ":constants",
        ":loops",
        ":slicing",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
    ],
)

cc_library(
    name = "quantize",
    hdrs = ["quantize.h"],
    deps = [
        ":constants",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "self_adjoint_eig",
    srcs = ["self_adjoint_eig.cc"],
    hdrs = ["self_adjoint_eig.h"],
    deps = [
        ":arithmetic",
        ":comparators",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":slicing",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "svd",
    srcs = ["svd.cc"],
    hdrs = ["svd.h"],
    deps = [
        ":arithmetic",
        ":comparators",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":slicing",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "tridiagonal",
    srcs = ["tridiagonal.cc"],
    hdrs = ["tridiagonal.h"],
    deps = [
        ":constants",
        ":loops",
        ":slicing",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:xla_builder",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "logdet",
    srcs = ["logdet.cc"],
    hdrs = ["logdet.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":qr",
        ":slicing",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/utils:common_utils",
    ],
)
