load("//third_party:repo.bzl", "tf_http_archive", "third_party_http_archive")
load("//third_party/build_option:dpcpp_configure.bzl", "dpcpp_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//third_party/llvm_project:setup.bzl", "llvm_setup")

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
        name = "onednn_cpu",
        # v2.7-rc
        tag = "v2.7-rc",
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
        name = "com_google_absl",
        build_file = "//third_party/absl:com_google_absl.BUILD",
        # TODO(itex): Remove the patch when https://github.com/abseil/abseil-cpp/issues/326 is resolved
        patch_file = "//third_party/absl:com_google_absl_fix_mac_and_nvcc_build.patch",
        sha256 = "35f22ef5cb286f09954b7cc4c85b5a3f6221c9d4df6b8c4a1e9d399555b366ee",
        strip_prefix = "abseil-cpp-997aaf3a28308eba1b9156aa35ab7bca9688e9f6",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
        ],
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

    new_git_repository(
        name = "onednn_gpu",
        # v2.7
        commit = "650085b2f3643aad05c629425983491d63b5c289",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn:onednn_gpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "onednn_graph",
        # llga public dev-graph-beta-2 branch
        commit = "19a62f17aff079536f54af6282d987e9c641840c",
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

    _itex_bind()
