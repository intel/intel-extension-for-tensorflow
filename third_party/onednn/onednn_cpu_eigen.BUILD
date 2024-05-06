exports_files(["LICENSE"])

load(
    "@intel_extension_for_tensorflow//third_party:common.bzl",
    "template_rule",
)
load(
    "@intel_extension_for_tensorflow//third_party/onednn:onednn.bzl",
    "gen_onednn_version",
)
load("@intel_extension_for_tensorflow//itex:itex.bzl", "if_cc_build")

_DNNL_CPU_COMMON = {
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "#undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
    "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
    "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
    "#cmakedefine DNNL_SYCL_HIP": "#undef DNNL_SYCL_HIP",
    "#cmakedefine DNNL_ENABLE_STACK_CHECKER": "#undef DNNL_ENABLE_STACK_CHECKER",
    "#cmakedefine DNNL_EXPERIMENTAL": "#undef DNNL_EXPERIMENTAL",
    "#cmakedefine01 BUILD_TRAINING": "#define BUILD_TRAINING 1",
    "#cmakedefine01 BUILD_INFERENCE": "#define BUILD_INFERENCE 0",
    "#cmakedefine01 BUILD_PRIMITIVE_ALL": "#define BUILD_PRIMITIVE_ALL 1",
    "#cmakedefine01 BUILD_BATCH_NORMALIZATION": "#define BUILD_BATCH_NORMALIZATION 0",
    "#cmakedefine01 BUILD_BINARY": "#define BUILD_BINARY 0",
    "#cmakedefine01 BUILD_CONCAT": "#define BUILD_CONCAT 0",
    "#cmakedefine01 BUILD_CONVOLUTION": "#define BUILD_CONVOLUTION 0",
    "#cmakedefine01 BUILD_DECONVOLUTION": "#define BUILD_DECONVOLUTION 0",
    "#cmakedefine01 BUILD_ELTWISE": "#define BUILD_ELTWISE 0",
    "#cmakedefine01 BUILD_INNER_PRODUCT": "#define BUILD_INNER_PRODUCT 0",
    "#cmakedefine01 BUILD_LAYER_NORMALIZATION": "#define BUILD_LAYER_NORMALIZATION 0",
    "#cmakedefine01 BUILD_LRN": "#define BUILD_LRN 0",
    "#cmakedefine01 BUILD_MATMUL": "#define BUILD_MATMUL 0",
    "#cmakedefine01 BUILD_POOLING": "#define BUILD_POOLING 0",
    "#cmakedefine01 BUILD_PRELU": "#define BUILD_PRELU 0",
    "#cmakedefine01 BUILD_REDUCTION": "#define BUILD_REDUCTION 0",
    "#cmakedefine01 BUILD_REORDER": "#define BUILD_REORDER 0",
    "#cmakedefine01 BUILD_RESAMPLING": "#define BUILD_RESAMPLING 0",
    "#cmakedefine01 BUILD_RNN": "#define BUILD_RNN 0",
    "#cmakedefine01 BUILD_SHUFFLE": "#define BUILD_SHUFFLE 0",
    "#cmakedefine01 BUILD_SOFTMAX": "#define BUILD_SOFTMAX 0",
    "#cmakedefine01 BUILD_SUM": "#define BUILD_SUM 0",
    "#cmakedefine01 BUILD_PRIMITIVE_CPU_ISA_ALL": "#define BUILD_PRIMITIVE_CPU_ISA_ALL 1",
    "#cmakedefine01 BUILD_SSE41": "#define BUILD_SSE41 0",
    "#cmakedefine01 BUILD_AVX2": "#define BUILD_AVX2 0",
    "#cmakedefine01 BUILD_AVX512": "#define BUILD_AVX512 0",
    "#cmakedefine01 BUILD_AMX": "#define BUILD_AMX 0",
    "#cmakedefine01 BUILD_PRIMITIVE_GPU_ISA_ALL": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 1",
    "#cmakedefine01 BUILD_GEN9": "#define BUILD_GEN9 0",
    "#cmakedefine01 BUILD_GEN11": "#define BUILD_GEN11 0",
    "#cmakedefine01 BUILD_XELP": "#define BUILD_XELP 0",
    "#cmakedefine01 BUILD_XEHPG": "#define BUILD_XEHPG 0",
    "#cmakedefine01 BUILD_XEHPC": "#define BUILD_XEHPC 0",
    "#cmakedefine01 BUILD_XEHP": "#define BUILD_XEHP 0",
    "#cmakedefine01 BUILD_GROUP_NORMALIZATION": "#define BUILD_GROUP_NORMALIZATION 1",
    "#cmakedefine01 BUILD_GEMM_KERNELS_ALL": "#define BUILD_GEMM_KERNELS_ALL 1",
    "#cmakedefine01 BUILD_GEMM_KERNELS_NONE": "#define BUILD_GEMM_KERNELS_NONE 0",
    "#cmakedefine01 BUILD_GEMM_SSE41": "#define BUILD_GEMM_SSE41 0",
    "#cmakedefine01 BUILD_GEMM_AVX2": "#define BUILD_GEMM_AVX2 0",
    "#cmakedefine01 BUILD_GEMM_AVX512": "#define BUILD_GEMM_AVX512 0",
    "#cmakedefine01 BUILD_XE2": "#define BUILD_XE2 0",
}

_DNNL_RUNTIME_TBB = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_TBB",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_TBB",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

_DNNL_RUNTIME_THREADPOOL = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

_DNNL_ONEDNN_GRAPH = {
    "#cmakedefine ONEDNN_BUILD_GRAPH": "#define ONEDNN_BUILD_GRAPH",
}

_DNNL_NO_ONEDNN_GRAPH = {
    "#cmakedefine ONEDNN_BUILD_GRAPH": "#undef ONEDNN_BUILD_GRAPH",
}

_TBB_WITH_ONEDNN_GRAPH_LIST = {}

_TBB_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_CPU_COMMON)

_TBB_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_RUNTIME_TBB)

_TBB_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_ONEDNN_GRAPH)

_OMP_WITH_ONEDNN_GRAPH_LIST = {}

_OMP_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_CPU_COMMON)

_OMP_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_RUNTIME_OMP)

_OMP_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_ONEDNN_GRAPH)

_OMP_WITHOUT_ONEDNN_GRAPH_LIST = {}

_OMP_WITHOUT_ONEDNN_GRAPH_LIST.update(_DNNL_CPU_COMMON)

_OMP_WITHOUT_ONEDNN_GRAPH_LIST.update(_DNNL_RUNTIME_OMP)

_OMP_WITHOUT_ONEDNN_GRAPH_LIST.update(_DNNL_NO_ONEDNN_GRAPH)

_THREADPOOL_WITH_ONEDNN_GRAPH_LIST = {}

_THREADPOOL_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_CPU_COMMON)

_THREADPOOL_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_RUNTIME_THREADPOOL)

_THREADPOOL_WITH_ONEDNN_GRAPH_LIST.update(_DNNL_ONEDNN_GRAPH)

# tbb + no llga build is not supported here, to simplify the logic here.
# TODO(itex): try better bazel usage in configuring strings with different options
template_rule(
    name = "dnnl_config_h",
    src = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = _THREADPOOL_WITH_ONEDNN_GRAPH_LIST,
)

# Create the file dnnl_version.h with DNNL version numbers.
gen_onednn_version(
    name = "onednn_version_generator",
    header_in = "include/oneapi/dnnl/dnnl_version.h.in",
    header_out = "include/oneapi/dnnl/dnnl_version.h",
)

_COPTS_LIST = [
    "-fexceptions",
    # TODO(itex): for symbol collision, may be removed in produce version
    "-fopenmp",
    "-Wno-unknown-pragmas",
] + [
    "-UUSE_MKL",
    "-UUSE_CBLAS",
    "-DDNNL_ENABLE_MAX_CPU_ISA",
    "-DDNNL_ENABLE_PRIMITIVE_CACHE",
] + if_cc_build(["-fvisibility=hidden"])

_INCLUDES_LIST = [
    "include",
    "src",
    "src/common",
    "src/common/ittnotify",
    "src/cpu",
    "src/cpu/gemm",
    "src/cpu/x64/xbyak",
]

_TEXTUAL_HDRS_LIST = glob(
    [
        "include/**/*",
    ],
) + [
    ":dnnl_config_h",
    ":onednn_version_generator",
]

cc_library(
    name = "onednn_libs_linux",
    srcs = [
        "@llvm_openmp//:libiomp5.so",
    ],
    hdrs = ["@llvm_openmp//:config_omp"],
    visibility = ["//visibility:public"],
)

# headers for onednn
cc_library(
    name = "onednn_cpu",
    copts = _COPTS_LIST,
    includes = ["include"],
    linkopts = ["-lrt"],
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
    deps = if_cc_build([
        ":onednn_cpu_lib",
        ":onednn_libs_linux",
    ]),
)

cc_library(
    name = "onednn_cpu_lib",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
            "src/common/ittnotify/*.c",
            "src/cpu/jit_utils/**/*.cpp",
            "src/cpu/x64/**/*.cpp",
        ],
        exclude = [
            "src/cpu/aarch64/**",
            "src/cpu/rv64/**",
            "src/graph/**",
        ],
    ),
    copts = _COPTS_LIST,
    includes = _INCLUDES_LIST,
    linkopts = ["-lrt"],
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
    alwayslink = True,
)

load("@intel_extension_for_tensorflow//third_party/onednn:build_defs.bzl", "if_graph_compiler", "if_llga_debug")

# TODO(itex): add graph compiler srcs, headers & check build option, once it is merged to oneDNN master.

_GRAPH_COPTS_CPU_LIST = [
    "-Wall",
    "-Wno-unknown-pragmas",
    "-fPIC",
    "-Wno-sign-compare",
    "-DBUILD_GRAPH",
    "-DDNNL_ENABLE_GRAPH_DUMP",
] + if_llga_debug([
    "-DDNNL_GRAPH_LAYOUT_DEBUG",
]) + if_graph_compiler([
    "-DDNNL_ENABLE_COMPILER_BACKEND",
    "-DSC_BUILTIN_JIT_ENABLED=1",
    "-DSC_CFAKE_JIT_ENABLED=1",
    "-DSC_LLVM_BACKEND=16",
    "-fopenmp",
]) + if_cc_build([
    "-fvisibility-inlines-hidden",
    "-fvisibility=hidden",
]) + select({
    "@intel_extension_for_tensorflow//third_party/onednn:build_with_tbb": ["-DSC_CPU_THREADPOOL=2"],
    "//conditions:default": ["-DSC_CPU_THREADPOOL=1"],
})

_GRAPH_SRCS_LIST = glob(
    [
        "src/graph/interface/*.cpp",
        "src/graph/backend/*.cpp",
        "src/graph/backend/dnnl/*.cpp",
        "src/graph/backend/fake/*.cpp",
        "src/graph/backend/dnnl/passes/*.cpp",
        "src/graph/backend/dnnl/patterns/*.cpp",
        "src/graph/backend/dnnl/kernels/*.cpp",
        "src/graph/utils/*.cpp",
        "src/graph/utils/pm/*.cpp",
    ],
) + if_graph_compiler(
    glob(
        [
            "src/graph/backend/graph_compiler/**/*.cpp",
        ],
        exclude = ["src/graph/backend/graph_compiler/core/src/compiler/jit/llvm/llvm_jit_resolver.cpp"],
    ),
)

_GRAPH_HDRS_LIST = glob(
    [
        "include/oneapi/dnnl/*",
    ],
)

_GRAPH_INCLUDES_LIST = [
    "include",
    "src/graph",
] + if_graph_compiler([
    "src/graph/backend/graph_compiler/core/",
    "src/graph/backend/graph_compiler/core/src/",
    "src/graph/backend/graph_compiler/core/src/compiler/",
])

_GRAPH_DEPS_LIST = [
    ":onednn_cpu",
] + if_graph_compiler(
    [
        "@llvm-project-16//llvm:Core",
        "@llvm-project-16//llvm:Support",
        "@llvm-project-16//llvm:Target",
        "@llvm-project-16//llvm:ExecutionEngine",
        "@llvm-project-16//llvm:MCJIT",
        "@llvm-project-16//llvm:X86CodeGen",
        "@llvm-project-16//llvm:AsmParser",
        "@llvm-project-16//llvm:AllTargetsAsmParsers",
    ],
)

# headers for onednn
cc_library(
    name = "onednn_graph_cpu",
    visibility = ["//visibility:public"],
    deps = [":onednn_cpu"] + if_cc_build([":onednn_graph_cpu_lib"]),
    alwayslink = True,
)

cc_library(
    name = "onednn_graph_cpu_lib",
    srcs = _GRAPH_SRCS_LIST,
    hdrs = _GRAPH_HDRS_LIST + glob(
        [
            "src/cpu/x64/xbyak/xbyak.h",
            "src/cpu/x64/xbyak/xbyak_mnemonic.h",
            "src/cpu/x64/xbyak/xbyak_util.h",
        ],
    ),
    # TODO(itex): find better way to include xbyak.h within onednn
    copts = _GRAPH_COPTS_CPU_LIST + ["-I external/onednn_cpu_eigen/src/cpu/x64"],
    includes = _GRAPH_INCLUDES_LIST,
    visibility = ["//visibility:public"],
    deps = _GRAPH_DEPS_LIST + if_graph_compiler([":onednn_graph_cpu_special_lib"]) + [":onednn_cpu_lib"],
    alwayslink = True,
)

cc_library(
    name = "onednn_graph_cpu_special_lib",
    srcs = ["src/graph/backend/graph_compiler/core/src/compiler/jit/llvm/llvm_jit_resolver.cpp"],
    hdrs = _GRAPH_HDRS_LIST,
    copts = _GRAPH_COPTS_CPU_LIST + [
        "-fno-rtti",  # special build option for llvm_jit_resolver.cpp
    ],
    includes = _GRAPH_INCLUDES_LIST,
    visibility = ["//visibility:public"],
    deps = _GRAPH_DEPS_LIST + [":onednn_cpu_lib"],
    alwayslink = True,
)
