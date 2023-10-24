load("//itex:itex.bzl", "cc_library", "itex_xpu_library")
load(
    "//itex/core/utils:build_config.bzl",
    "tf_protobuf_deps",
)
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")

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

filegroup(
    name = "resource_hdrs",
    srcs = [
        "casts.h",
        "errors.h",
        "hash.h",
        "intrusive_ptr.h",
        "managed_stack_trace.h",
        "mutex.h",
        "mutex_data.h",
        "numbers.h",
        "platform_types.h",
        "protobuf.h",
        "refcount.h",
        "resource_base.h",
        "resource_handle.h",
        "stack_frame.h",
        "status.h",
        "statusor.h",
        "statusor_internals.h",
        "str_util.h",
        "strcat.h",
        "tensor_coding.h",
        "tensor_shape.h",
        "thread_annotations.h",
        "type_index.h",
    ],
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
    name = "ctstring",
    hdrs = [
        "ctstring.h",
        "ctstring_internal.h",
    ],
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
        exclude = [
            "ctstring.h",
            "ctstring_internal.h",
        ],
    ),
    linkstatic = 1,
    deps = [
        ":device_gpu_impl",
        "//itex/core/graph:config_util_hdr",
        "//itex/core/utils/gtl:gtl_libs",
        "//itex/core/utils/tensor_bundle:byteswaparray",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_googlesource_code_re2//:re2",
        "@double_conversion//:double-conversion",
        "@farmhash_archive//:farmhash",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
        "@nsync//:nsync_cpp",
    ],
)

cc_library(
    name = "device_gpu_impl",
    deps = if_dpcpp([
        "//itex/core/devices/gpu:eigen_stream_device",
        "//itex/core/devices/gpu:gpu_device_impl",
    ]),
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
        ":resource_hdrs",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "device_types",
    srcs = [
        "device_types.cc",
    ],
    hdrs = [
        "device_types.h",
        "macros.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    hdrs = [
        "device_types.h",
        "types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":tstring",
        "@local_config_tf//:protos_all",
    ],
)

cc_library(
    name = "logging",
    srcs = [
        "env_time.cc",
        "logging.cc",
    ],
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
    name = "cpu_info",
    srcs = ["cpu_info.cc"],
    hdrs = [
        "cpu_info.h",
        "integral_types.h",
        "platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        "@com_google_absl//absl/base",
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
    name = "byte_order",
    hdrs = ["byte_order.h"],
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    deps = [
        ":logging",
        ":platform",
        ":status",
        ":str_util",
        ":strcat",
        ":types",
        "@com_google_absl//absl/strings",
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
    name = "statusor",
    srcs = [
        "statusor.cc",
        "statusor_internals.h",
    ],
    hdrs = ["statusor.h"],
    deps = [
        ":errors",
        ":logging",
        ":status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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

itex_xpu_library(
    name = "hw_info",
    srcs = ["hw_info.cc"],
    hdrs = ["hw_info.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hw_info_hdr",
    ],
)

cc_library(
    name = "hw_info_hdr",
    srcs = ["hw_info.h"],
    hdrs = ["macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/build_option/dpcpp:itex_gpu_header",
    ],
)

cc_library(
    name = "parallel_openmp",
    hdrs = [
        "parallel_openmp.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "parallel",
    hdrs = [
        "parallel.h",
    ],
    visibility = ["//visibility:public"],
)
