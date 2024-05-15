# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.

licenses([
    # Note: Eigen is an MPL2 library that includes GPL v3 and LGPL v2.1+ code.
    #       We've taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["LICENSE"])

load("@intel_extension_for_tensorflow//itex:itex.bzl", "cc_library", "if_gpu_backend")

#load("//third_party/mkl:build_defs.bzl", "if_mkl")

EIGEN3_THIRD_PARTY_HEADERS = [
    "Eigen/Core",
    "Eigen/LU",
    "Eigen/Cholesky",
    "Eigen/Eigenvalues",
    "Eigen/OrderingMethods",
    "Eigen/QR",
    "Eigen/SparseCholesky",
    "Eigen/SparseCore",
    "Eigen/SVD",
    "unsupported/Eigen/MatrixFunctions",
    "unsupported/Eigen/SpecialFunctions",
    "unsupported/Eigen/CXX11/ThreadPool",
    "unsupported/Eigen/CXX11/Tensor",
    "unsupported/Eigen/CXX11/FixedPoint",
] + glob(["unsupported/Eigen/CXX11/src/FixedPoint/*.h"])

cc_library(
    name = "eigen3",
    hdrs = EIGEN3_THIRD_PARTY_HEADERS,
    includes = [],  #+ if_mkl(["./mkl_include"]),
    visibility = ["//visibility:public"],
    deps = [
        "@itex_eigen_archive//:eigen",
    ] + if_gpu_backend([
        "//third_party/build_option/sycl:eigen_itex_gpu_header",
    ]),
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = ["**/OWNERS"],
    ),
    visibility = ["//itex:__subpackages__"],
)

filegroup(
    name = "eigen_third_party_header_files",
    srcs = EIGEN3_THIRD_PARTY_HEADERS,
    visibility = ["//visibility:public"],
)

genrule(
    name = "install_eigen_headers",
    srcs = [
        "@itex_eigen_archive//:eigen_header_files",
        ":eigen_third_party_header_files",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#*external/itex_eigen_archive/}"

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
)
