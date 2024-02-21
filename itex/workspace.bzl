load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/build_option:dpcpp_configure.bzl", "dpcpp_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//third_party/llvm_project:setup.bzl", "llvm_setup")
load("//third_party/llvm_project:setup_16.bzl", "llvm_setup_16")
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

    farmhash()
    absl()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
        ],
    )

    tf_http_archive(
        name = "pybind11",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.10.0.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.10.0.tar.gz",
        ],
        sha256 = "eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
        strip_prefix = "pybind11-2.10.0",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
        system_build_file = clean_dep("//third_party/systemlibs:pybind11.BUILD"),
    )

    # v3.3
    _ONEDNN_CPU_COMMIT = "08fea71aff4c273e34579e86396405f95d34aa74"

    new_git_repository(
        name = "onednn_cpu",
        commit = _ONEDNN_CPU_COMMIT,
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn:onednn_cpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "onednn_cpu_eigen",
        commit = _ONEDNN_CPU_COMMIT,
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn:onednn_cpu_eigen.BUILD"),
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
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"),
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
        name = "rules_python",
        sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
            "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
        ],
    )

    http_archive(
        name = "rules_pkg",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
        ],
        sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["//third_party/protobuf:protobuf.patch"],
        #build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
        strip_prefix = "protobuf-3.21.9",
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.21.9.zip",
            "https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip",
        ],
    )

    tf_http_archive(
        name = "nsync",
        build_file = clean_dep("//third_party:nsync.BUILD"),
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
        name = "onednn_gpu",
        # rls-v3.4
        commit = "ea4d952",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = clean_dep("//third_party/onednn:onednn_gpu.BUILD"),
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    # llvm built in bazel
    llvm_setup(name = "itex-llvm-project")

    #llvm built in bazel
    llvm_setup_16(name = "llvm-project-16")

    EIGEN_COMMIT = "d10b27fe37736d2944630ecd7557cefa95cf87c9"
    tf_http_archive(
        name = "itex_eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        patch_file = ["//third_party/eigen3:intel_ext.patch"],
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

    new_git_repository(
        name = "itex_xetla",
        commit =
            "28c18c4c1ff5aa247b9750413ff61b60edc2c1cb",
        remote =
            "https://github.com/intel/xetla.git",
        verbose = True,
        build_file = "//third_party/xetla:BUILD",
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    _itex_bind()
