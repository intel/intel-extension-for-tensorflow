# Description: CTC, Connectionist Temporal Classification,
# is a type of seq2seq loss.  The libraries in this directory
# implement the CTC loss and a number of CTC decoders.
load("//itex:itex.bzl", "cc_library")

cc_library(
    name = "ctc_loss_calculator_lib",
    srcs = [
        "ctc_loss_calculator.cc",
    ],
    hdrs = [
        "ctc_loss_calculator.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":ctc_loss_util_lib",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "ctc_loss_util_lib",
    hdrs = [
        "ctc_loss_util.h",
    ],
)
