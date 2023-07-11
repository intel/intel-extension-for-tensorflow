package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xetla_header",
    hdrs = glob(
        include = [
            "include/common/*/*.h",
            "include/group/*.h",
            "include/group/brgemm/*/*.h",
            "include/group/epilogue/*/*.h",
            "include/kernel/*.h",
            "include/kernel/gemm/*/*.h",
            "include/subgroup/*.h",
            "include/subgroup/tile/*/*.h",
            "include/experimental/*.h",
            "include/experimental/common/*.h",
            "include/experimental/group/*/*.h",
            "include/experimental/kernel/*/*.h",
            "include/experimental/subgroup/*/*.h",
        ],
    ),
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
    ],
)
