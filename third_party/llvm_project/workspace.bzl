"""Provides the repository macro to import LLVM."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("//third_party:repo.bzl", "tf_http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo_13(name):
    """Imports LLVM."""
    LLVM_COMMIT = "75e33f71c2dae584b13a7d1186ae0a038ba98838"
    LLVM_SHA256 = "9e2ef2fac7525a77220742a3384cafe7a35adc7e5c9750378b2cf25c2d2933f5"

    tf_http_archive(
        name = name,
        # branch llvmorg-13.0.1
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm_project:llvm.BUILD",
        patch_file = ["//third_party/llvm_project:simplify_llvm_build.patch"],
    )

def repo(name):
    """Imports LLVM."""

    LLVM_COMMIT = "ac1ec9e2904a696e360b40572c3b3c29d67981ef"

    # TF commit: 850f6f7e58deb4d947f2542842459b7a7021d2c0
    LLVM_SHA256 = "3a7cd0be43916c6d860d08e46f084eae8035a248e87097725e95d4a966119d93"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm_project:llvm.BUILD",
        patch_file = [
            "//third_party/llvm_project:spirv.patch",
            "//third_party/llvm_project:llvm_build.patch",
            "//third_party/llvm_project:mlir.patch",
        ],
    )
