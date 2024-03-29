"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@llvm-raw-16//utils/bazel:configure.bzl", "llvm_configure")

# The subset of LLVM targets that TensorFlow cares about.
_LLVM_TARGETS = [
    "X86",
]

def llvm_setup_16(name):
    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(
        name = name,
        repo_mapping = {"@python_runtime": "@local_config_python"},
        targets = _LLVM_TARGETS,
    )
