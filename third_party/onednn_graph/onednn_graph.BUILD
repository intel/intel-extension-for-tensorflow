load("@intel_extension_for_tensorflow//third_party/onednn_graph:build_defs.bzl", "if_graph_compiler", "if_llga_debug")

_COPTS_CPU_LIST = [
    "-Wall",
    "-Wno-unknown-pragmas",
    "-fvisibility-inlines-hidden",
    "-fPIC",
    "-fvisibility=hidden",
    "-Wno-sign-compare",
    "-DDNNL_GRAPH_ENABLE_DUMP",
] + if_llga_debug([
    "-DDNNL_GRAPH_LAYOUT_DEBUG",
]) + if_graph_compiler([
    "-DDNNL_GRAPH_ENABLE_COMPILER_BACKEND",
    "-DSC_LLVM_BACKEND=13",
    "-DSC_CPU_THREADPOOL=1",
    "-fopenmp",
])

_COPTS_GPU_LIST = [
    "-DDNNL_GRAPH_ENABLE_COMPILED_PARTITION_CACHE",
    "-DDNNL_GRAPH_GPU_RUNTIME=512",
    # 512 corresponding to "DPCPP" , https://github.com/intel-innersource/libraries.performance.math.onednn/blob/4b33801238e9179e27ec5c5b4d2d9585936f15c4/CMakeLists.txt#L213-L215
    "-DDNNL_GRAPH_WITH_SYCL",
    "-DDNNL_GRAPH_GPU_SYCL",
    "-DDNNL_GRAPH_ENABLE_DUMP",
    "-Wall",
    "-Wno-unknown-pragmas",
    "-Wno-deprecated-declarations",
    "-ffp-model=precise",
    "-fno-reciprocal-math",
    "-fvisibility-inlines-hidden",
    "-Wno-sign-compare",
    "-Wno-pass-failed",
    "-Wno-tautological-compare",
    "-fPIC",
    "-fvisibility=hidden",
    "-dpcpp_compile",
] + if_llga_debug([
    "-DDNNL_GRAPH_LAYOUT_DEBUG",
])

_SRCS_LIST = glob(
    [
        "src/interface/*.cpp",
        "src/backend/*.cpp",
        "src/backend/dnnl/*.cpp",
        "src/backend/fake/*.cpp",
        "src/backend/dnnl/passes/*.cpp",
        "src/backend/dnnl/patterns/*.cpp",
        "src/backend/dnnl/kernels/*.cpp",
        "src/utils/*.cpp",
        "src/utils/pm/*.cpp",
    ],
) + if_graph_compiler(
    glob(
        [
            "src/backend/graph_compiler/*.cpp",
            "src/backend/graph_compiler/core/src/*.cpp",
            "src/backend/graph_compiler/core/src/**/*.cpp",
            "src/backend/graph_compiler/core/src/**/**/*.cpp",
            "src/backend/graph_compiler/core/src/**/**/**/*.cpp",
            "src/backend/graph_compiler/patterns/*.cpp",
            "src/backend/graph_compiler/patterns/*.hpp",
        ],
        exclude = ["src/backend/graph_compiler/core/src/compiler/jit/llvm/llvm_jit_resolver.cpp"],
    ),
)

_HDRS_LIST = glob(
    [
        "include/*",
        "include/llga/*",
        "include/oneapi/dnnl/*",
        "src/interface/*.hpp",
        "src/backend/*.hpp",
        "src/backend/dnnl/*.hpp",
        "src/backend/fake/*.hpp",
        "src/backend/dnnl/passes/*.hpp",
        "src/backend/dnnl/patterns/*.hpp",
        "src/backend/dnnl/kernels/*.hpp",
        "src/utils/*.hpp",
        "src/utils/pm/*.hpp",
    ],
) + if_graph_compiler(glob([
    "src/backend/graph_compiler/*.hpp",
    "src/backend/graph_compiler/core/src/*.hpp",
    "src/backend/graph_compiler/core/src/**/*.hpp",
    "src/backend/graph_compiler/core/src/**/**/*.hpp",
    "src/backend/graph_compiler/core/src/**/**/**/*.hpp",
]))

_INCLUDES_LIST = [
    "include",
    "src",
] + if_graph_compiler([
    "src/backend/graph_compiler/core/",
    "src/backend/graph_compiler/core/src/",
    "src/backend/graph_compiler/core/src/compiler/",
])

_DEPS_LIST = [
    "@onednn_cpu_v2//:onednn_cpu",
] + if_graph_compiler(
    [
        "@llvm-project-13//llvm:Core",
        "@llvm-project-13//llvm:Support",
        "@llvm-project-13//llvm:Target",
        "@llvm-project-13//llvm:ExecutionEngine",
        "@llvm-project-13//llvm:MCJIT",
        "@llvm-project-13//llvm:X86CodeGen",
        "@llvm-project-13//llvm:AsmParser",
        "@llvm-project-13//llvm:AllTargetsAsmParsers",
    ],
)

cc_library(
    name = "onednn_graph_cpu",
    srcs = _SRCS_LIST,
    hdrs = _HDRS_LIST,
    # TODO(itex): find better way to include xbyak.h within onednn
    copts = _COPTS_CPU_LIST + ["-I external/onednn_cpu_v2/src/cpu/x64"],
    includes = _INCLUDES_LIST,
    visibility = ["//visibility:public"],
    deps = _DEPS_LIST + if_graph_compiler([":onednn_graph_cpu_special"]),
    alwayslink = True,
)

# graph compiler special kernel
cc_library(
    name = "onednn_graph_cpu_special",
    srcs = ["src/backend/graph_compiler/core/src/compiler/jit/llvm/llvm_jit_resolver.cpp"],
    hdrs = _HDRS_LIST,
    copts = _COPTS_CPU_LIST + [
        "-fno-rtti",  # special build option for llvm_jit_resolver.cpp
    ] + ["-I external/onednn_cpu_v2/src/cpu/x64"],
    includes = _INCLUDES_LIST,
    visibility = ["//visibility:public"],
    deps = _DEPS_LIST,
)

cc_library(
    name = "onednn_graph_gpu",
    srcs = _SRCS_LIST,
    hdrs = _HDRS_LIST,
    copts = _COPTS_GPU_LIST,
    includes = _INCLUDES_LIST,
    visibility = ["//visibility:public"],
    deps = ["@onednn_gpu_v2//:onednn_gpu"],
    alwayslink = True,
)
