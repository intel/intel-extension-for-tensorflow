package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gpu_profiler",
    srcs = ["gpu_profiler.cc"],
    hdrs = ["gpu_collector.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":ze_tracer",
        "//itex/core:protos_all_cc",
        "//itex/core/profiler/utils:xplane_utils",
        "//itex/core/utils:common_utils",
        "//itex/core/utils:logging",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "ze_tracer",
    srcs = [
        "trace_options.h",
        "tracing.h",
        "utils.h",
        "ze_api_collector.h",
        "ze_kernel_collector.h",
        "ze_tracer.h",
        "ze_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [":ze_correlator"],
)

cc_library(
    name = "ze_correlator",
    srcs = ["correlator.cc"],
    hdrs = [
        "correlator.h",
        "pti_assert.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
    ],
)
