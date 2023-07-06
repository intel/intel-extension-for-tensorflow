load(
    "//itex/core/utils:build_config.bzl",
    "tf_proto_library",
)
load(
    "//itex/python:build_defs.bzl",
    "gen_itex_version",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

# Export the BUILD file so automated tooling can check licenses
exports_files([
    "BUILD",
])

tf_proto_library(
    name = "protos_all",
    srcs = [],
    cc_api_version = 2,
    protodeps = [
        "//itex/core/utils/protobuf:for_core_protos",
    ],
    visibility = ["//visibility:public"],
)

# Create the file itex_version.h with real version numbers.
gen_itex_version(
    name = "itex_version_generator",
    header_in = ":version.h.in",
    header_out = ":version.h",
    visibility = ["//visibility:public"],
)
