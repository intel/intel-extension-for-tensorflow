package(default_visibility = ["//visibility:public"])

cc_library(
    name = "graphcycles",
    srcs = ["graphcycles.cc"],
    hdrs = ["graphcycles.h"],
    deps = [
        ":ordered_set",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ordered_set",
    hdrs = ["ordered_set.h"],
    deps = [
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)
