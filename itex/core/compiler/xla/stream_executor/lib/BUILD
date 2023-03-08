package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "lib",
    srcs = glob(
        [
            "**/*.cc",
        ],
        exclude = [
            "**/*test*",
        ],
    ),
    hdrs = glob(["**/*.h"]),
    deps = [
        "//itex/core:protos_all_cc",
        "//itex/core/compiler/xla/stream_executor/platform",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)
