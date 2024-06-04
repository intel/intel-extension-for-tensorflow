"""Build rules for pybind11"""

load("@itex_local_config_sycl//sycl:build_defs.bzl", "if_sycl")

PYBIND_COPTS = [
    "-fexceptions",
]

PYBIND_FEATURES = [
    "-use_header_modules",
    "-parse_headers",
]

PYBIND_DEPS = [
    "@pybind11",
]

def clean_dep(dep):
    return str(Label(dep))

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level + ["tensorflow-plugins"]))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    levels_to_root = native.package_name().count("/") + name.count("/")
    return [
        "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
    ]

def pybind_extension(
        name,
        copts = [],
        features = [],
        linkopts = [],
        tags = [],
        deps = [],
        **kwargs):
    tags = tags + ["req_dep=%s" % dep for dep in PYBIND_DEPS]
    native.cc_binary(
        name = name + ".so",
        copts = copts + PYBIND_COPTS + ["-fvisibility=hidden"],
        features = features + PYBIND_FEATURES,
        linkopts = ["-Wl,-Bsymbolic"] + linkopts + ["-Wl,-rpath,$$ORIGIN/../"] + if_sycl(["-lsycl"]),
        linkshared = 1,
        tags = tags + ["local"],
        deps = deps + PYBIND_DEPS,
        **kwargs
    )

def pybind_library(
        name,
        copts = [],
        features = [],
        tags = [],
        deps = [],
        **kwargs):
    tags = tags + ["req_dep=%s" % dep for dep in PYBIND_DEPS]

    native.cc_library(
        name = name,
        copts = copts + PYBIND_COPTS,
        features = features + PYBIND_FEATURES,
        tags = tags,
        deps = deps + PYBIND_DEPS,
        **kwargs
    )

def gen_itex_version(name, header_in, header_out, **kwargs):
    tool = "//itex/python:gen_itex_version"

    native.genrule(
        name = name,
        srcs = [header_in],
        outs = [header_out],
        tools = [tool],
        cmd = "$(location {}) ".format(tool) + "--in=$< " + "--out=$@",
        stamp = True,
        **kwargs
    )
