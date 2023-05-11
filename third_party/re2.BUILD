# Copyright 2009 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Bazel (http://bazel.io/) BUILD file for RE2.

licenses(["notice"])

exports_files(["LICENSE"])

load("@intel_extension_for_tensorflow//itex:itex.bzl", "cc_library")

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
)

cc_library(
    name = "re2",
    srcs = [
        "re2/bitmap256.h",
        "re2/bitstate.cc",
        "re2/compile.cc",
        "re2/dfa.cc",
        "re2/filtered_re2.cc",
        "re2/mimics_pcre.cc",
        "re2/nfa.cc",
        "re2/onepass.cc",
        "re2/parse.cc",
        "re2/perl_groups.cc",
        "re2/prefilter.cc",
        "re2/prefilter.h",
        "re2/prefilter_tree.cc",
        "re2/prefilter_tree.h",
        "re2/prog.cc",
        "re2/prog.h",
        "re2/re2.cc",
        "re2/regexp.cc",
        "re2/regexp.h",
        "re2/set.cc",
        "re2/simplify.cc",
        "re2/tostring.cc",
        "re2/unicode_casefold.cc",
        "re2/unicode_casefold.h",
        "re2/unicode_groups.cc",
        "re2/unicode_groups.h",
        "re2/walker-inl.h",
        "util/flags.h",
        "util/logging.h",
        "util/pod_array.h",
        "util/rune.cc",
        "util/sparse_array.h",
        "util/sparse_set.h",
        "util/strutil.cc",
        "util/strutil.h",
        "util/utf.h",
        "util/util.h",
    ],
    hdrs = [
        "re2/filtered_re2.h",
        "re2/re2.h",
        "re2/set.h",
    ],
    copts = select({
        ":windows": [],
        ":windows_msvc": [],
        "//conditions:default": ["-pthread"],
    }),
    linkopts = select({
        # Darwin doesn't need `-pthread' when linking and it appears that
        # older versions of Clang will warn about the unused command line
        # argument, so just don't pass it.
        ":darwin": [],
        ":windows": [],
        ":windows_msvc": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)
