load("//itex:itex.bzl", "itex_xpu_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

itex_xpu_library(
    name = "exact_uniform_int",
    hdrs = ["exact_uniform_int.h"],
)

itex_xpu_library(
    name = "philox",
    srcs = [
        "distribution_sampler.cc",
        "random_distributions.cc",
        "simple_philox.cc",
    ],
    hdrs = [
        "distribution_sampler.h",
        "random_distributions.h",
        "simple_philox.h",
    ],
    deps = [
        ":exact_uniform_int",
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
        ":random",
        "//itex/core/utils:common_utils",
    ],
)

itex_xpu_library(
    name = "random",
    hdrs = ["random.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//itex/core/utils:random",
    ],
)

itex_xpu_library(
    name = "weighted_picker",
    srcs = ["weighted_picker.cc"],
    hdrs = ["weighted_picker.h"],
    deps = [
        ":philox",
        "//itex/core/utils:common_utils",
    ],
    alwayslink = 1,
)
