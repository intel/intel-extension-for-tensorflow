load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/build_option:dpcpp_configure.bzl", "dpcpp_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//third_party/llvm_project:setup.bzl", "llvm_setup")
load("//third_party/llvm_project:setup_13.bzl", "llvm_setup_13")
load("//third_party:tf_runtime/workspace.bzl", tf_runtime = "repo")
load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load(
    "//third_party/farmhash:workspace.bzl",
    farmhash = "repo",
)
load(
    "//third_party/absl:workspace.bzl",
    absl = "repo",
)

def clean_dep(dep):
    return str(Label(dep))

def _itex_bind():
    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = str(Label("//third_party/python_runtime:headers")),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def itex_workspace(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds"""
    dpcpp_configure(name = "local_config_dpcpp")
    syslibs_configure(name = "local_config_syslibs")
    tf_runtime()
    stablehlo()
    farmhash()
    absl()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "109a99384f9d08f9e75136d218ebaebc68cc810c56897aea2224c57932052d30",
        strip_prefix = "bazel-toolchains-94d31935a2c94fe7e7c7379a0f3393e181928ff7",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/94d31935a2c94fe7e7c7379a0f3393e181928ff7.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/94d31935a2c94fe7e7c7379a0f3393e181928ff7.tar.gz",
        ],
    )

    tf_http_archive(
        name = "pybind11",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
        ],
        sha256 = "90b705137b69ee3b5fc655eaca66d0dc9862ea1759226f7ccd3098425ae69571",
        strip_prefix = "pybind11-2.6.0",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
        system_build_file = clean_dep("//third_party/systemlibs:pybind11.BUILD"),
    )

    new_git_repository(
        name = "onednn_cpu_v2",
        # Align to SPR gold release.
        commit = "b1ea77cdb7468ca334d50dbc19f72aed44435507",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn_v2:onednn_cpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "onednn_cpu",
        # rls-v3.1
        commit = "ad34c124895690bafd2b110577639824899ecbca",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn:onednn_cpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    # OneDNN cpu backend with TBB runtime.
    git_repository(
        name = "oneTBB",
        tag = "v2021.5.0",
        remote = "https://github.com/oneapi-src/oneTBB/",
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9",
        strip_prefix = "re2-506cfa4bffd060c06ec338ce50ea3468daa6c814",
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
            "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        ],
    )

    tf_http_archive(
        name = "double_conversion",
        build_file = clean_dep("//third_party:double_conversion.BUILD"),
        sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
        strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
        system_build_file = clean_dep("//third_party/systemlibs:double_conversion.BUILD"),
        urls = [
            "http://mirror.tensorflow.org/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
            "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        ],
    )

    tf_http_archive(
        name = "zlib",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
        strip_prefix = "zlib-1.2.13",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.13.tar.gz",
            "https://zlib.net/zlib-1.2.13.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = clean_dep("//third_party/protobuf:protobuf.patch"),
        #build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
        strip_prefix = "protobuf-3.9.2",
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
            "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        ],
    )

    tf_http_archive(
        name = "nsync",
        sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
        strip_prefix = "nsync-1.22.0",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
            "https://github.com/google/nsync/archive/1.22.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "rules_python",
        sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz"),
    )

    new_git_repository(
        name = "onednn_gpu_v2",
        commit = "5c7d2549efd4cde805931ef3214ffebff5ef1d1c",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn_v2:onednn_gpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "onednn_gpu",
        # rls-v3.1
        commit = "ad34c124895690bafd2b110577639824899ecbca",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn:onednn_gpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "onednn_graph",
        # llga public dev-graph-beta-3 branch
        commit = "147d9bcec306738be5f223028b181e0ba592caf7",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn_graph:onednn_graph.BUILD"),
        verbose = True,
        patch_args = ["-p1"],
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    # llvm built in bazel
    llvm_setup(name = "llvm-project")

    # llvm built in bazel
    llvm_setup_13(name = "llvm-project-13")

    EIGEN_COMMIT = "d10b27fe37736d2944630ecd7557cefa95cf87c9"
    tf_http_archive(
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        patch_file = clean_dep("//third_party/eigen3:intel_ext.patch"),
        sha256 = "a3c10a8c14f55e9f09f98b0a0ac6874c21bda91f65b7469d9b1f6925990e867b",
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
    )

    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = clean_dep("//third_party/llvm_openmp:BUILD"),
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz",
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz",
        ],
    )

    tf_http_archive(
        name = "six_archive",
        build_file = "//third_party:six.BUILD",
        sha256 = "30639c035cdb23534cd4aa2dd52c3bf48f06e5f4a941509c8bafd8ce11080259",
        strip_prefix = "six-1.15.0",
        system_build_file = "//third_party/systemlibs:six.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
        ],
    )

    git_repository(
        name = "mlir-hlo",
        commit = "1b862a645b61b954c3353bca3469e51f3f3b1ca7",
        remote = "https://github.com/tensorflow/mlir-hlo/",
        verbose = True,
        patches = ["//third_party/mlir_hlo:mlir_hlo.patch"],
        patch_args = ["-p1"],
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    git_repository(
        name = "spir_headers",
        commit = "93754d52d6cbbfd61f4e87571079e8a28e65f8ca",
        remote = "https://github.com/KhronosGroup/SPIRV-Headers.git",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "llvm_spir",
        commit = "a3b372cb2d0250fbd5e395c3d32613f1644dfeb5",
        remote = "https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git",
        build_file = "//third_party/llvm_spir:llvm_spir.BUILD",
        verbose = True,
        patches = ["//third_party/llvm_spir:llvm_spir.patch"],
        patch_args = ["-p1"],
    )

    _itex_bind()
