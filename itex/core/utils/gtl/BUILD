cc_library(
    name = "gtl_libs",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:prefetch",
        "//itex/core/utils:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
