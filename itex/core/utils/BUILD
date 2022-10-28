load(
    "//itex/core/utils:build_config.bzl",
    "tf_protobuf_deps",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "common_util_hdrs",
    srcs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        ":common_utils",
    ],
)

cc_library(
    name = "tf_buffer",
    srcs = ["tf_buffer.cc"],
    hdrs = [
        "tf_buffer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + tf_protobuf_deps(),
)

cc_library(
    name = "common_utils",
    srcs = glob(
        [
            "*.cc",
        ],
    ),
    hdrs = glob(
        [
            "*.h",
        ],
    ),
    linkstatic = 1,
    deps = [
        "//itex/core/utils/gtl:gtl_libs",
        "//third_party/eigen3",
        "//itex/core/devices:xpu_device_util",
        "@double_conversion//:double-conversion",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/container:inlined_vector",
        "@nsync//:nsync_cpp",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//itex/core/devices/gpu:eigen_stream_device",
            "//itex/core/devices/gpu:gpu_device_impl",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    visibility = ["//visibility:public"],
    deps = [":platform"],
)

cc_library(
    name = "tstring",
    hdrs = [
        "tstring.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":tstring",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = [
        "env_time.h",
        "integral_types.h",
        "logging.h",
        "macros.h",
        "numeric_types.h",
        "stringpiece.h",
        "types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":tstring",
        "//itex/core/utils/gtl:gtl_libs",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "mutex",
    srcs = ["mutex.cc"],
    hdrs = [
        "mutex.h",
        "mutex_data.h",
        "thread_annotations.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        "@nsync//:nsync_cpp",
    ],
)

cc_library(
    name = "str_util",
    srcs = ["str_util.cc"],
    hdrs = [
        "str_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":platform",
        ":types",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "stringprintf",
    srcs = ["stringprintf.cc"],
    hdrs = [
        "stringprintf.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":types",
    ],
)

cc_library(
    name = "strcat",
    srcs = ["strcat.cc"],
    hdrs = [
        "platform_types.h",
        "strcat.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":numbers",
        ":platform",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "numbers",
    srcs = ["numbers.cc"],
    hdrs = [
        "numbers.h",
        "platform_types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":platform",
        ":str_util",
        ":stringprintf",
        ":types",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@double_conversion//:double-conversion",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "abi",
    srcs = ["abi.cc"],
    hdrs = [
        "abi.h",
        "integral_types.h",
        "platform_types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = [
        "stacktrace.h",
        "status.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":abi",
        ":logging",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf_version",
    hdrs = ["tf_version.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "env_var",
    srcs = ["env_var.cc"],
    hdrs = [
        "env_var.h",
        "errors.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":numbers",
        ":status",
        ":str_util",
        ":strcat",
        ":tf_version",
        ":types",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "hw_info",
    srcs = ["hw_info.cc"],
    hdrs = [
        "hw_info.h",
        "macros.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/build_option/dpcpp:dpcpp_header",
    ],
)
