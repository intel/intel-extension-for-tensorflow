load("//itex:itex.bzl", "itex_xpu_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

itex_xpu_library(
    name = "philox",
    srcs = [
        "random_distributions.cc",
    ],
    hdrs = [
        "random_distributions.h",
    ],
    deps = [
        ":philox_random",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

itex_xpu_library(
    name = "philox_random",
    hdrs = ["philox_random.h"],
    deps = [
        "//itex/core/utils:common_utils",
    ],
)

itex_xpu_library(
    name = "guarded_philox_random",
    srcs = ["guarded_philox_random.cc"],
    hdrs = [
        "guarded_philox_random.h",
        "philox_random.h",
    ],
    deps = [
        "//itex/core/utils:common_utils",
    ],
)
