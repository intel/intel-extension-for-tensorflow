workspace(name = "intel_extension_for_tensorflow")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//third_party:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.1.0")

load("//itex:tf_configure.bzl", "tf_configure")
load("//third_party/py:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

tf_configure(name = "local_config_tf")

load("//itex:workspace1.bzl", "itex_workspace1")

itex_workspace1()

load("//itex:workspace.bzl", "clean_dep", "itex_workspace")

itex_workspace()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()
