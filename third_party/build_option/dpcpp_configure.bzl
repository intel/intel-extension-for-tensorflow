# -*- Python -*-
"""DPCPP autoconfiguration.
`dpcpp_configure` depends on the following environment variables:

  * HOST_CXX_COMPILER:  The host C++ compiler
  * HOST_C_COMPILER:    The host C compiler
  * PYTHON_LIB_PATH: The path to the python lib
"""

_HOST_CXX_COMPILER = "HOST_CXX_COMPILER"

_HOST_C_COMPILER = "HOST_C_COMPILER"

_DPCPP_TOOLKIT_PATH = "DPCPP_TOOLKIT_PATH"

_DPCPP_COMPILER_VERSION = "DPCPP_COMPILER_VERSION"

_ONEAPI_MKL_PATH = "ONEAPI_MKL_PATH"

_TF_NEED_MKL = "TF_NEED_MKL"

_AOT_CONFIG = "AOT_CONFIG"

_PYTHON_LIB_PATH = "PYTHON_LIB_PATH"

_TF_SHARED_LIBRARY_DIR = "TF_SHARED_LIBRARY_DIR"

_PYTHON_LIB_DIR = "PYTHON_LIB_DIR"

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _enable_dpcpp(repository_ctx):
    if "TF_NEED_DPCPP" in repository_ctx.os.environ:
        enable_dpcpp = repository_ctx.os.environ["TF_NEED_DPCPP"].strip()
        return enable_dpcpp == "1"
    return False

def _enable_mkl(repository_ctx):
    if _TF_NEED_MKL in repository_ctx.os.environ:
        enable_mkl = repository_ctx.os.environ[_TF_NEED_MKL].strip()
        return enable_mkl == "1"
    return False

def _enable_dpcpp_build(repository_ctx):
    return _DPCPP_TOOLKIT_PATH in repository_ctx.os.environ

def auto_configure_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def find_c(repository_ctx):
    """Find host C compiler."""
    c_name = "gcc"
    if _HOST_C_COMPILER in repository_ctx.os.environ:
        c_name = repository_ctx.os.environ[_HOST_C_COMPILER].strip()
    if c_name.startswith("/"):
        return c_name
    c = repository_ctx.which(c_name)
    if c == None:
        fail("Cannot find C compiler, please correct your path.")
    return c

def find_cc(repository_ctx):
    """Find host C++ compiler."""
    cc_name = "g++"
    if _HOST_CXX_COMPILER in repository_ctx.os.environ:
        cc_name = repository_ctx.os.environ[_HOST_CXX_COMPILER].strip()
    if cc_name.startswith("/"):
        return cc_name
    cc = repository_ctx.which(cc_name)
    if cc == None:
        fail("Cannot find C++ compiler, please correct your path.")
    return cc

def find_dpcpp_root(repository_ctx):
    """Find DPC++ compiler."""
    sycl_name = ""
    if _DPCPP_TOOLKIT_PATH in repository_ctx.os.environ:
        sycl_name = repository_ctx.os.environ[_DPCPP_TOOLKIT_PATH].strip()
    if sycl_name.startswith("/"):
        return sycl_name
    fail("Cannot find DPC++ compiler, please correct your path")

def get_dpcpp_version(repository_ctx):
    """Get DPC++ compiler version yyyymmdd"""
    default_version = "00000000"
    macro = "__INTEL_LLVM_COMPILER"
    version_file = "include/sycl/CL/sycl/version.hpp"
    base_path = find_dpcpp_root(repository_ctx)
    intel_llvm_macro = "00000000"
    compiler_bin_path = base_path + "/bin/icpx"
    compiler_macros = repository_ctx.execute([compiler_bin_path, "-dM", "-E", "-xc++", "/dev/null"])
    macro_list = compiler_macros.stdout.split("\n")
    for m in macro_list:
        result = m.strip().split(" ")
        if macro in result:
            intel_llvm_macro = result[-1]
    if intel_llvm_macro >= "20230000":
        version_file = "include/sycl/version.hpp"
    full_path = repository_ctx.path(base_path + "/" + version_file)
    if not full_path.exists:
        return default_version
    f = repository_ctx.read(full_path)
    lines = str(f).split("\n")
    for l in lines:
        if l.startswith("#define"):
            l_list = l.strip().split(" ")
            if (l_list[0] == "#define" and
                l_list[1] == "__SYCL_COMPILER_VERSION"):
                default_version = l_list[-1]
    return default_version

def find_mkl_path(repository_ctx):
    """Find MKL Path."""
    mkl_path = ""
    if _ONEAPI_MKL_PATH in repository_ctx.os.environ:
        mkl_path = repository_ctx.os.environ[_ONEAPI_MKL_PATH].strip()
    if mkl_path.startswith("/"):
        return mkl_path
    fail("Cannot find OneAPI MKL, please correct your path")

def find_aot_config(repository_ctx):
    """Find AOT config."""
    aot_config = " -Xs \'-options -cl-poison-unsupported-fp64-kernels\'"
    device_tmp = " -fsycl-targets=spir64_gen,spir64 -Xs \'-device {}\'"
    if _AOT_CONFIG in repository_ctx.os.environ:
        devices = repository_ctx.os.environ[_AOT_CONFIG].strip()
        device_list = []
        if devices:
            device_list = devices.split(",")
        if device_list:
            # check for security purpose only here
            for d in device_list:
                if len(d) > 20:
                    fail("Invalid AOT target: {}".format(d))
            aot_config += device_tmp.format(devices)
    return aot_config

def find_python_lib(repository_ctx):
    """Returns python path."""
    if _PYTHON_LIB_PATH in repository_ctx.os.environ:
        return repository_ctx.os.environ[_PYTHON_LIB_PATH].strip()
    fail("Environment variable PYTHON_LIB_PATH was not specified re-run ./configure")

def _check_lib(repository_ctx, toolkit_path, lib):
    """Checks if lib exists under sycl_toolkit_path or fail if it doesn't.

    Args:
      repository_ctx: The repository context.
      toolkit_path: The toolkit directory containing the libraries.
      ib: The library to look for under toolkit_path.
    """
    lib_path = toolkit_path + "/" + lib
    if not repository_ctx.path(lib_path).exists:
        auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
    """Checks whether the directory exists and fail if it does not.

    Args:
      repository_ctx: The repository context.
      directory: The directory to check the existence of.
    """
    if not repository_ctx.path(directory).exists:
        auto_configure_fail("Cannot find dir: %s" % directory)

def _symlink_dir(repository_ctx, src_dir, dest_dir):
    """Symlinks all the files in a directory.

    Args:
      repository_ctx: The repository context.
      src_dir: The source directory.
      dest_dir: The destination directory to create the symlinks in.
    """
    files = repository_ctx.path(src_dir).readdir()
    for src_file in files:
        repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        Label("//third_party/build_option/%s.tpl" % tpl),
        substitutions,
    )

def _file(repository_ctx, label):
    repository_ctx.template(
        label.replace(":", "/"),
        Label("//third_party/build_option/%s" % label),
        {},
    )

_INC_DIR_MARKER_BEGIN = "#include <...>"

# OSX add " (framework directory)" at the end of line, strip it.
_OSX_FRAMEWORK_SUFFIX = " (framework directory)"

_OSX_FRAMEWORK_SUFFIX_LEN = len(_OSX_FRAMEWORK_SUFFIX)

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
    if path.endswith(_OSX_FRAMEWORK_SUFFIX):
        path = path[:-_OSX_FRAMEWORK_SUFFIX_LEN].strip()
    return path

def _normalize_include_path(repository_ctx, path):
    """Normalizes include paths before writing them to the crosstool.

      If path points inside the 'crosstool' folder of the repository, a relative
      path is returned.
      If path points outside the 'crosstool' folder, an absolute path is returned.
      """
    path = str(repository_ctx.path(path))
    crosstool_folder = str(repository_ctx.path(".").get_child("crosstool"))

    if path.startswith(crosstool_folder):
        # We drop the path to "$REPO/crosstool" and a trailing path separator.
        return "\"" + path[len(crosstool_folder) + 1:] + "\""
    return "\"" + path + "\""

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    result = repository_ctx.execute([cc, "-E", "-x" + lang, "-", "-v"])
    index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = result.stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = result.stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = result.stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = result.stderr[index1 + 1:]
    else:
        inc_dirs = result.stderr[index1 + 1:index2].strip()

    return [
        _normalize_include_path(repository_ctx, _cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
    ]

def get_cxx_inc_directories(repository_ctx, cc):
    """Compute the list of default C and C++ include directories."""

    # For some reason `clang -xc` sometimes returns include paths that are
    # different from the ones from `clang -xc++`. (Symlink and a dir)
    # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)

    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp
    ]

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_sycl_disabled():
  fail("ERROR: Building with --config=dpcpp but TensorFlow is not configured " +
       "to build with DPCPP support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with DPCPP support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_sycl_disabled.bzl", "error_sycl_disabled")

error_sycl_disabled()
"""

def _create_dummy_repository(repository_ctx):
    # Set up BUILD file for sycl/.
    _tpl(repository_ctx, "dpcpp:build_defs.bzl")
    _tpl(repository_ctx, "dpcpp:BUILD")
    _tpl(repository_ctx, "dpcpp:platform.bzl")

    # If sycl_configure is not configured to build with SYCL support, and the user
    # attempts to build with --config=sycl, add a dummy build rule to intercept
    # this and fail with an actionable error message.
    repository_ctx.file(
        "crosstool/error_sycl_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

    _tpl(
        repository_ctx,
        "dpcpp:build_defs.bzl",
        {
            "%{dpcpp_is_configured}": "False",
            "%{dpcpp_build_is_configured}": "False",
            "%{mkl_is_configured}": "False",
        },
    )

def _sycl_autoconf_imp(repository_ctx):
    """Implementation of the sycl_autoconf rule."""
    builtin_include_dirs = ""
    unfiltered_cxx_flags = ""
    linker_flags = ""

    dpcpp_defines = {}

    if not _enable_dpcpp(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        # copy template files
        _tpl(repository_ctx, "dpcpp:build_defs.bzl")
        _tpl(repository_ctx, "dpcpp:BUILD")
        _tpl(repository_ctx, "dpcpp:platform.bzl")

        additional_cxxflags = []
        additional_linker_flags = []
        builtin_includes = []

        builtin_includes += [find_dpcpp_root(repository_ctx) + "/include"]
        builtin_includes += [find_dpcpp_root(repository_ctx) + "/lib/clang/12.0.0/include"]
        builtin_includes += [find_dpcpp_root(repository_ctx) + "/lib/clang/13.0.0/include"]

        pwd = repository_ctx.os.environ["PWD"]
        additional_inc = []
        if repository_ctx.os.environ.get("CPATH") != None:
            for p in repository_ctx.os.environ["CPATH"].strip().split(":"):
                if p != "":
                    additional_inc += [_normalize_include_path(repository_ctx, p)]
        if len(additional_inc) > 0:
            additional_inc = ",".join(additional_inc)
        else:
            additional_inc = "\"\""

        if _enable_mkl(repository_ctx) and repository_ctx.os.environ.get("ONEAPI_MKL_PATH") != None:
            dpcpp_defines["%{ONEAPI_MKL_PATH}"] = str(find_mkl_path(repository_ctx))
            builtin_includes += [find_mkl_path(repository_ctx) + "/include"]
        else:
            dpcpp_defines["%{ONEAPI_MKL_PATH}"] = ""
        if repository_ctx.os.environ.get("TMPDIR") != None:
            dpcpp_defines["%{TMP_DIRECTORY}"] = repository_ctx.os.environ.get("TMPDIR")
        else:
            tmp_suffix = repository_ctx.execute(["cat", "/proc/sys/kernel/random/uuid"]).stdout.rstrip()
            tmp_dir = "/tmp/" + tmp_suffix
            dpcpp_defines["%{TMP_DIRECTORY}"] = tmp_dir

        dpcpp_defines["%{cxx_builtin_include_directories}"] = str(builtin_includes)
        dpcpp_defines["%{dpcpp_builtin_include_directories}"] = str(builtin_includes)
        dpcpp_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""
        dpcpp_defines["%{unfiltered_compile_flags}"] = ""
        dpcpp_defines["%{host_compiler}"] = "gcc"
        dpcpp_defines["%{HOST_COMPILER_PATH}"] = "/usr/bin/gcc"
        dpcpp_defines["%{host_compiler_prefix}"] = "/usr/bin"
        dpcpp_defines["%{dpcpp_compiler_root}"] = str(find_dpcpp_root(repository_ctx))
        dpcpp_defines["%{linker_bin_path}"] = "/usr/bin"
        dpcpp_defines["%{DPCPP_ROOT_DIR}"] = str(find_dpcpp_root(repository_ctx))
        dpcpp_defines["%{AOT_DEVICES}"] = str(find_aot_config(repository_ctx))
        dpcpp_defines["%{TF_NEED_MKL}"] = repository_ctx.os.environ[_TF_NEED_MKL].strip()
        dpcpp_defines["%{DPCPP_RUNTIME_INC}"] = pwd + "/third_party/build_option/dpcpp/runtime/"
        dpcpp_defines["%{additional_include_directories}"] = additional_inc
        dpcpp_defines["%{TF_SHARED_LIBRARY_DIR}"] = repository_ctx.os.environ[_TF_SHARED_LIBRARY_DIR]
        dpcpp_defines["%{DPCPP_COMPILER_VERSION}"] = str(get_dpcpp_version(repository_ctx))
        dpcpp_defines["%{PYTHON_LIB_PATH}"] = repository_ctx.os.environ[_PYTHON_LIB_PATH]

        unfiltered_cxx_flags = "" if additional_cxxflags == [] else "unfiltered_cxx_flag: "
        unfiltered_cxx_flags += "\n  unfiltered_cxx_flag: ".join(additional_cxxflags)

        dpcpp_defines["%{unfiltered_compile_flags}"] = unfiltered_cxx_flags

        linker_flags = "" if additional_linker_flags == [] else "linker_flag: "
        linker_flags += "\n  linker_flag: ".join(additional_linker_flags)

        _tpl(repository_ctx, "crosstool_dpcpp:BUILD", dpcpp_defines)
        _tpl(repository_ctx, "crosstool_dpcpp/bin:crosstool_wrapper_driver", dpcpp_defines)
        _tpl(repository_ctx, "crosstool_dpcpp:cc_toolchain_config.bzl", dpcpp_defines)

        if _enable_dpcpp_build(repository_ctx):
            dpcpp_build_defines = {}
            dpcpp_build_defines["%{dpcpp_is_configured}"] = "True"
            dpcpp_build_defines["%{dpcpp_build_is_configured}"] = "True"
            if _enable_mkl(repository_ctx):
                dpcpp_build_defines["%{mkl_is_configured}"] = "True"
            dpcpp_root = find_dpcpp_root(repository_ctx)
            _check_dir(repository_ctx, dpcpp_root)

            _tpl(
                repository_ctx,
                "dpcpp:build_defs.bzl",
                dpcpp_build_defines,
            )

dpcpp_configure = repository_rule(
    local = True,
    implementation = _sycl_autoconf_imp,
)
