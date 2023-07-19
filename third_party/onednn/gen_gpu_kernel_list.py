#!/usr/bin/env python3

import re
import os
import sys

is_v2 = False

class Kernel(object):
  def __init__(self, path):
    self.extern_ = self._parse_extern(path)
    self.kernels_ = self._parse_kernels(path, self.extern_)
    self.path_ = path

  def extern(self):
    extern_prefix = "extern const char *"
    extern_suffix = "_kernel;"
    if not is_v2:
      extern_suffix =  "_kernel[];"
    return extern_prefix + self.extern_ + extern_suffix

  def entries(self):
    format_extern = lambda x: '        {{ "{}", {}_kernel }},'.format(
        x, self.extern_)
    return [format_extern(entry) for entry in self.kernels_]

  def content(self, inc_dirs):
    path = os.path.basename(self.path_)
    kernel_name, _ = os.path.splitext(path)
    return kernel_name, '\n'.join(self._extend_includes(inc_dirs, self.path_))

  def subfolder(self, sub):
    kernel_dir = os.path.dirname(self.path_)
    index = kernel_dir.rfind(sub)
    index += len(sub) + 1  # step to the last if possible
    if index == len(kernel_dir):
      return ""
    else:
      return kernel_dir[index:]

  def _parse_extern(self, path):
    path = os.path.basename(path)
    file_name, _ = os.path.splitext(path)
    return file_name

  def _parse_kernels(self, path, extern_name):
    with open(path) as f:
      content = f.read()
      pattern = 'kernel[ \n]+void[ \n]+([a-z0-9_]+)'
      kernels = re.findall(pattern, content, re.DOTALL)
      return kernels

  def _extend_includes(self, inc_dirs, path):
    ret = []
    pattern = re.compile('^\\s*#include "(.*)"')
    with open(path) as f:
      lines = f.readlines()
      for line in lines:
        result = pattern.match(line)
        if result is not None:
          inc_file = result.group(1)
          for inc_dir in inc_dirs:
            inc_path = os.path.join(inc_dir, inc_file)
            if not os.path.exists(inc_path):
              continue
            inc_lines = self._extend_includes(inc_dirs, inc_path)
            ret.extend(inc_lines)
            break
        else:
          if is_v2:
            line = line.strip()
            line = line.replace("\n", "")
            quoted_line = 'R"==({})==""\\n"'.format(line)
          else:
            line = line.replace('\\', '\\\\')
            line = line.replace('"', '\\"')
            line = line.replace("\n", "\\n")
            quoted_line = '"{}",'.format(line)
          ret.append(quoted_line)

    return ret

class Header(Kernel):
  def __init__(self, path):
    self.extern_ = self._parse_extern(path)
    self.kernels_ = self._parse_kernels(path, self.extern_)
    self.path_ = path

  def extern(self):
    extern_prefix = "extern const char *"
    extern_suffix = "_header;"
    return extern_prefix + self.extern_ + extern_suffix

  def values(self):
    format_entry = lambda x: '        {}_header,'.format(x)
    return [format_entry(self.extern_)]

  def entries(self):
    dir_path = os.path.dirname(self.path_)
    src_index = dir_path.rfind("src")
    header_path = self.path_[src_index + len("src") : ]
    print(header_path)
    print(self.extern_)
    return '        {{ "{}", {}_header }},'.format(header_path, self.extern_)

  def name(self):
    dir_path = os.path.dirname(self.path_)
    src_index = dir_path.rfind("src")
    header_path = self.path_[src_index + len("src") : ]
    return '        "{}",'.format(header_path)

class KernelList(object):
  def __init__(self, folder, header_dir):
    self.kernels_ = []
    self.headers_ = []

    cl_files = self._get_cl_suffix_files(folder)
    for clf in cl_files:
      self.kernels_.append(Kernel(clf))

    if is_v2:
      header_files = self._get_header_files(header_dir)
      for hf in header_files:
        self.headers_.append(Header(hf))

  def generate_list(self, src, target):
    externs = []
    entries = ['\n']  # for style

    for kernel in self.kernels_:
      externs.append(kernel.extern())
      entries.extend(kernel.entries())

    externs_content = '\n'.join(externs)
    entries_content = '\n'.join(entries)

    if is_v2:
      header_externs = []
      header_values = ['\n']
      header_names = ['\n']
      header_entries = ['\n']
      for header in self.headers_:
        header_externs.append(header.extern())
        header_values.extend(header.values())
        header_names.append(header.name())
        header_entries.append(header.entries())

      header_externs_content = "\n".join(header_externs)
      header_values_content = "\n".join(header_values)
      header_names_content = "\n".join(header_names)
      header_entries_content = "\n".join(header_entries)

    with open(src) as f:
      content = f.read()
      content = content.replace('@KER_LIST_EXTERN@', externs_content)
      content = content.replace('@KER_LIST_ENTRIES@', entries_content)
      if is_v2:
        content = content.replace('@KER_HEADERS_EXTERN@', header_externs_content)
        content = content.replace('@KER_HEADERS@', header_values_content)
        content = content.replace('@KER_HEADER_NAMES@', header_names_content)
        content = content.replace('@KER_HEADER_LIST_ENTRIES@', header_entries_content)

    with open(target, 'w') as f:
      f.write(content)

  def _generate(self, inc_dirs, root, sub, impl, suffix):
    impl_name, content = impl.content(inc_dirs)
    more_sub = impl.subfolder(sub)
    # construct the file xxx_suffix.cpp
    file_name = impl_name + "_" + suffix + ".cpp"

    target = os.path.join(root, sub, more_sub)

    if not os.path.exists(target):
      os.makedirs(target, exist_ok=True)

    with open(os.path.join(target, file_name), "w") as f:
      f.write(self.format(impl_name, suffix, content))

  def generate_kernel(self, inc_dirs, root, sub):
    """
        It will expand all codes and convert it into string including the new line char.

        For compatible with bazel, we will create a folder structure which is the same
        with original source. Such as,

        src/gpu/ocl/xxxx_kernel.cpp
        src/gpu/ocl/gemm/xxxx_kernel.cpp

        which is not the same with cmake generated, because it will generate all kernel
        files into src/gpu/ocl/, no subfolder.
        """
    for kernel in self.kernels_:
      self._generate(inc_dirs, root, sub, kernel, "kernel")

  def generate_header(self, inc_dirs, root, sub):
    for header in self.headers_:
      self._generate(inc_dirs, root, sub, header, "header")

  def format(self, name, suffix, content):
    """
    Format the content. Pay attention that, there's no comma before nullptr in header.
    """

    header = """
namespace dnnl {{
namespace impl {{
namespace gpu {{
namespace ocl {{
    const char* {}_{} =
{};
}}
}}
}}
}}
        """

    if not is_v2:
      header = """
namespace dnnl {{
namespace impl {{
namespace gpu {{
namespace ocl {{
    const char* {}_kernel[] = {{
{}
        nullptr
    }};
}}
}}
}}
}}
"""
      return header.format(name, content)

    return header.format(name, suffix, content)

  def _get_suffix_files(self, folder, suffix):
    files = []
    for root, _, filenames in os.walk(folder):
      for filename in sorted(filenames):
        s = os.path.splitext(filename)[-1]
        if s == suffix:
          f = os.path.join(root, filename)
          files.append(f)
    return files

  def _get_cl_suffix_files(self, folder):
    return self._get_suffix_files(folder, ".cl")

  def _get_header_files(self, folder):
    return self._get_suffix_files(folder, ".h")


class FilesHelper(object):
  def __init__(self, in_file, out_dir):
    """
        This class is a helper class which will matain the input and output files/dirs.

        There's an assumption, the ocl impls folder is "src/gpu/ocl". And also assuming
        the include directory is same depth with "src".

        If possible, it should passed as an arguments, but for covenience, we do this
        assumption directly.
        """
    OCL_IMPL_DIR = "src/gpu/ocl"
    HEADER_ROOT_DIR = "src/gpu"
    IN_FILE = "ocl_kernel_list.cpp.in"

    in_file = os.path.expanduser(in_file)
    out_dir = os.path.expanduser(out_dir)

    in_dir = os.path.dirname(in_file)
    in_dir = in_dir[:-len(OCL_IMPL_DIR)]

    self.inc_dirs = [os.path.join(in_dir, "src"), os.path.join(in_dir, "include")]
    self.ocl_impls_dir = os.path.join(in_dir, OCL_IMPL_DIR)
    self.gen_kernel_list_cpp_in = os.path.join(in_dir, OCL_IMPL_DIR, IN_FILE)

    self.out_root = out_dir
    self.out_subfolder = OCL_IMPL_DIR

    self.gen_kernel_list_cpp = os.path.join(out_dir, OCL_IMPL_DIR,
                                            os.path.splitext(IN_FILE)[0])

    kernels_out = os.path.join(self.out_root, self.out_subfolder)
    if not os.path.exists(kernels_out):
      os.makedirs(kernels_out)

    self.header_dir = os.path.join(in_dir, HEADER_ROOT_DIR)
    self.header_subfolder = HEADER_ROOT_DIR

def parse_args(argv):
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v

  return result

def enable_v2(in_file):
  with open(in_file, "r") as f:
    for line in f.readlines():
      if "KER_HEADERS" in line:
        global is_v2
        is_v2 = True
        break

def main():
  args = parse_args(sys.argv[1:])

  # The --in argument is the ocl_kernel_list.cpp.in file path.
  # The --out argument is the output folder
  files_helper = FilesHelper(args["--in"], args["--out"])

  only_gen_header=False
  if args["--header"].lower() == "true":
    only_gen_header=True

  enable_v2(files_helper.gen_kernel_list_cpp_in)

  kernel_list = KernelList(files_helper.ocl_impls_dir, files_helper.header_dir)
  if not only_gen_header:
    kernel_list.generate_list(files_helper.gen_kernel_list_cpp_in,
                              files_helper.gen_kernel_list_cpp)
    kernel_list.generate_kernel(files_helper.inc_dirs, files_helper.out_root,
                                files_helper.out_subfolder)
  if only_gen_header and is_v2:
    kernel_list.generate_header(files_helper.inc_dirs, files_helper.out_root,
                                files_helper.header_subfolder)


if __name__ == "__main__":
  main()
