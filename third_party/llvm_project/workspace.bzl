"""Provides the repository macro to import LLVM."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("//third_party:repo.bzl", "tf_http_archive")

def clean_dep(dep):
    return str(Label(dep))

# graph compiler requires llvm > 16.0.6, which contains necessary fix in https://github.com/llvm/llvm-project/commit/801dd8870fe3634e81e35e88519477541d1b0119
def repo_16(name):
    """Imports LLVM."""

    # v16.0.6
    LLVM_COMMIT = "7cbf1a2591520c2491aa35339f227775f4d3adf6"
    LLVM_SHA256 = "d6d78afbb3f8aea68e4517e6ebc8e8436a08a05502279debb5d3fad5b908243e"

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
            "//third_party/llvm_project:llvm.patch",
        ],
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
