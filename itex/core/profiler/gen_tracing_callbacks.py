# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#==============================================================
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: MIT
# =============================================================

import os
import sys
import re

# Parse Level Zero Headers ####################################################

STATE_NORMAL = 0
STATE_CONDITION = 1
STATE_SKIP = 2

def get_comma_count(line):
  count = 0
  level = 0
  for symbol in line:
    if symbol == "(":
      level += 1
    elif symbol == ")":
      assert level > 0
      level -= 1
    elif symbol == "," and level == 0:
      count += 1
  return count

def remove_comments(line):
  pos = line.find("//")
  if pos != -1:
    line = line[0:pos]
  return line

def get_func_name(callback_struct_name):
  assert callback_struct_name.strip().find("ze_pfn") == 0
  assert callback_struct_name.strip().find("Cb_t") + len("Cb_t") == len(callback_struct_name.strip())
  body = callback_struct_name.strip()
  body = body.split("ze_pfn")[1]
  body = body.split("Cb_t")[0]
  return "ze" + body

def get_struct_range(lines, struct_name):
  start = -1
  for i in range(len(lines)):
    if lines[i].find(struct_name) != -1:
      start = i
      break
  assert start >= 0

  while lines[start].find("{") == -1:
    start +=1
  start += 1
  assert start < len(lines)

  end = start
  while lines[end].find("}") == -1:
    end += 1
  assert end < len(lines)

  return start, end

def get_callback_struct_map(f, struct_name):
  f.seek(0)
  lines = f.readlines()

  start, end = get_struct_range(lines, struct_name)

  struct_map = {}
  cond = ""
  state = STATE_NORMAL
  for i in range(start, end):
    if lines[i].find("#if") >= 0:
      items = lines[i].strip().split()
      assert len(items) == 2
      state = STATE_CONDITION
      cond = items[1].strip()
      continue
    elif lines[i].find("#else") >= 0:
      assert state == STATE_CONDITION
      state = STATE_SKIP
      continue
    elif lines[i].find("#endif") >= 0:
      assert state != STATE_NORMAL
      state = STATE_NORMAL
      cond = ""
      continue

    if state == STATE_SKIP:
      continue

    items = lines[i].strip().split()
    assert len(items) == 2
    type_name = items[0].strip()
    field_name = items[1].strip().strip(";")
    assert not (type_name in struct_map)
    struct_map[type_name] = (field_name, cond)

  return struct_map

def get_params(f, func_name):
  f.seek(0)
  params = []

  param_struct_name = get_param_struct_name(func_name)
  lines = f.readlines()
  start, end = get_struct_range(lines, param_struct_name)

  for i in range(start, end):
    items = lines[i].strip().split()
    assert len(items) >= 2

    type = ""
    for j in range(len(items) - 1):
      type += items[j] + " "
    type = type.strip()
    assert type[len(type) - 1] == "*"
    type = type[0:len(type) - 1]

    name = items[len(items) - 1].rstrip(";")
    assert name[0] == "p"
    name = name[1:len(name)]

    assert not ((name, type) in params)
    params.append((name, type))

  return params

def find_enums(f, enum_map):
  f.seek(0)
  lines = f.readlines()

  enum_list = []
  for line in lines:
    if line.find("typedef enum") != -1:
      items = line.strip().rstrip("{").split()
      assert len(items) == 3
      enum_list.append(items[2].strip().lstrip("_"))

  for enum_name in enum_list:
    params = {}
    start, end = get_struct_range(lines, enum_name)
    default_value = 0
    has_unresolved_values = False
    for i in range(start, end):
      line = remove_comments(lines[i]).strip()
      if not line:
        continue
      comma_count = get_comma_count(line)
      assert comma_count == 0 or comma_count == 1
      if line.find("=") == -1:
        assert not has_unresolved_values
        field_name = line.rstrip(",")
        field_value = str(default_value)
        default_value += 1
      else:
        items = line.split("=")
        assert len(items) == 2
        field_name = items[0].strip()
        field_value = items[1].strip().rstrip(",")
        if field_value.find("0x") == 0:
          field_value = str(int(field_value, 16))
        if all(symbol.isdigit() for symbol in field_value):
          default_value = int(field_value)
          default_value += 1
        else:
          has_unresolved_values = True
      assert not (field_name in params)
      params[field_name] = field_value
    assert len(params) > 0
    assert not (enum_name in enum_map)
    enum_map[enum_name] = params

def get_param_struct_name(func_name):
  assert func_name[0] == 'z'
  func_name = 'Z' + func_name[1:]
  func_name = func_name.replace("CL", "Cl")
  func_name = func_name.replace("IPC", "Ipc")
  items = re.findall('[A-Z][^A-Z]*', func_name)
  assert len(items) > 1
  struct_name = ""
  for item in items:
    struct_name += item.lower() + "_"
  struct_name += "params_t"
  return struct_name

def get_func_list(f):
  f.seek(0)
  func_list = []
  for line in f.readlines():
    if line.find("ZE_APICALL") != -1 and line.find("ze_pfn") != -1:
      items = line.split("ze_pfn")
      assert len(items) == 2
      assert items[1].find("Cb_t") != -1
      items = items[1].split("Cb_t")
      assert len(items) == 2
      func_list.append("ze" + items[0].strip())
  return func_list

def get_callback_group_map(f):
  group_map = {}

  base_map = get_callback_struct_map(f, "ze_callbacks_t")
  assert len(base_map) > 0

  for key, value in base_map.items():
    func_map = get_callback_struct_map(f, key)
    for fkey, fvalue in func_map.items():
      func_name = get_func_name(fkey)
      assert not (func_name in group_map)
      group_map[func_name] = (value, fvalue)

  return group_map

def get_param_map(f):
  param_map = {}
  func_list = get_func_list(f)

  for func in func_list:
    assert not (func in param_map)
    param_map[func] = get_params(f, func)

  return param_map

def get_enum_map(include_path):
  enum_map = {}

  for file_name in os.listdir(include_path):
    if file_name.endswith(".h") or file_name.endswith(".hpp"):
      file_path = os.path.join(include_path, file_name)
      file = open(file_path, "rt")
      find_enums(file, enum_map)
      file.close()

  return enum_map

# Generate Callbacks ##########################################################

def gen_api(f, func_list, group_map):
  f.write("static void SetTracingAPIs(zel_tracer_handle_t tracer) {\n")
  f.write("  zet_core_callbacks_t prologue = {};\n")
  f.write("  zet_core_callbacks_t epilogue = {};\n")
  f.write("\n")
  for func in func_list:
    assert func in group_map
    group, callback = group_map[func]
    group_name = group[0]
    group_cond = group[1]
    assert not group_cond
    callback_name = callback[0]
    callback_cond = callback[1]
    if callback_cond:
      f.write("#if " + callback_cond + "\n")
    f.write("  prologue." + group_name + "." + callback_name + " = " + func + "OnEnter;\n")
    f.write("  epilogue." + group_name + "." + callback_name + " = " + func + "OnExit;\n")
    if callback_cond:
      f.write("#endif //" + callback_cond + "\n")
  f.write("\n")
  f.write("  ze_result_t status = ZE_RESULT_SUCCESS;\n")
  f.write("  status = zelTracerSetPrologues(tracer, &prologue);\n")
  f.write("  PTI_ASSERT(status == ZE_RESULT_SUCCESS);\n")
  f.write("  status = zelTracerSetEpilogues(tracer, &epilogue);\n")
  f.write("  PTI_ASSERT(status == ZE_RESULT_SUCCESS);\n")
  f.write("}\n")
  f.write("\n")

def gen_structure_type_converter(f, enum_map):
  struct_type_enum = {}
  for name in enum_map["ze_structure_type_t"]:
    struct_type_enum[name] = int(enum_map["ze_structure_type_t"][name])
  struct_type_enum = sorted(struct_type_enum.items(), key=lambda x:x[1])
  assert "ze_structure_type_t" in enum_map
  f.write("static const char* GetStructureTypeString(unsigned structure_type) {\n")
  f.write("  switch (structure_type) {\n")
  for name, value in struct_type_enum:
    f.write("    case " + name + ":\n")
    f.write("      return \"" + name + "\";\n")
  f.write("    default:\n")
  f.write("      break;\n")
  f.write("  }\n")
  f.write("  return \"UNKNOWN\";\n")
  f.write("}\n")
  f.write("\n")

def gen_result_converter(f, enum_map):
  result_enum = {}
  for name in enum_map["ze_result_t"]:
    result_enum[name] = int(enum_map["ze_result_t"][name])
  result_enum = sorted(result_enum.items(), key=lambda x:x[1])
  assert "ze_result_t" in enum_map
  f.write("static const char* GetResultString(unsigned result) {\n")
  f.write("  switch (result) {\n")
  for name, value in result_enum:
    f.write("    case " + name + ":\n")
    f.write("      return \"" + name + "\";\n")
  f.write("    default:\n")
  f.write("      break;\n")
  f.write("  }\n")
  f.write("  return \"UNKNOWN\";\n")
  f.write("}\n")
  f.write("\n")

def gen_enum(f, enum_map, enum_name, param_name):
  assert enum_name in enum_map
  f.write("    switch (" + param_name + ") {\n")
  for name, value in enum_map[enum_name].items():
    f.write("      case " + value + ":\n")
    f.write("        stream << \"" + name + "\";\n")
    f.write("        break;\n")
  f.write("      default:\n")
  f.write("        stream << \"<UNKNOWN>\";\n")
  f.write("        break;\n")
  f.write("    }\n")

def gen_enter_callback(f, func, params, enum_map):
  f.write("  PTI_ASSERT(global_user_data != nullptr);\n")
  f.write("  ZeApiCollector* collector =\n")
  f.write("    reinterpret_cast<ZeApiCollector*>(global_user_data);\n")
  f.write("  PTI_ASSERT(collector->correlator_ != nullptr);\n")
  f.write("\n")
  f.write("  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);\n")
  f.write("  start_time = collector->GetTimestamp();\n")
  f.write("  if (collector->options_.call_tracing) {\n")
  f.write("    std::stringstream stream;\n")
  f.write("    stream << \">>>> [\" << start_time << \"] \";\n")
  f.write("    if (collector->options_.need_pid) {\n")
  f.write("      stream << \"<PID:\" << utils::GetPid() << \"> \";\n")
  f.write("    }\n")
  f.write("    if (collector->options_.need_tid) {\n")
  f.write("      stream << \"<TID:\" << utils::GetTid() << \"> \";\n")
  f.write("    }\n")
  f.write("    stream << \"" + func + "\" << \":\";\n")
  for name, type in params:
    if type == "ze_ipc_mem_handle_t" or type == "ze_ipc_event_pool_handle_t":
      f.write("    stream << \" " + name + " = \" << (params->p" + name + ")->data;\n")
    else:
      if type.find("char*") >= 0 and type.find("char*") == len(type) - len("char*"):
        f.write("    if (*(params->p" + name + ") == nullptr) {\n")
        f.write("      stream << \" " + name + " = \" << \"0\";\n")
        f.write("    } else if (strlen(*(params->p" + name +")) == 0) {\n")
        f.write("      stream << \" " + name + " = \\\"\\\"\";\n")
        f.write("    } else {\n")
        f.write("      stream << \" " + name + " = \\\"\" << *(params->p" + name + ") << \"\\\"\";\n")
        f.write("    }\n")
      else:
        f.write("    stream << \" " + name + " = \" << *(params->p" + name + ");\n")
        if name.find("ph") == 0 or name.find("pptr") == 0 or name.find("pCount") == 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          if type == "ze_ipc_mem_handle_t*" or type == "ze_ipc_event_pool_handle_t*":
            f.write("      stream << \" (" + name[1:] + " = \" << (*(params->p" + name + "))->data << \")\";\n")
          else:
            f.write("      stream << \" (" + name[1:] + " = \" << **(params->p" + name + ") << \")\";\n")
          f.write("    }\n")
        elif type.find("ze_group_count_t*") >= 0:
          f.write("    if (*(params->p" + name +") != nullptr) {\n")
          f.write("      stream << \" {\" << (*(params->p" + name + "))->groupCountX << \", \";\n")
          f.write("      stream << (*(params->p" + name + "))->groupCountY << \", \";\n")
          f.write("      stream << (*(params->p" + name + "))->groupCountZ << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_event_pool_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->count << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_command_queue_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->ordinal << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->index << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->mode << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->priority << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_kernel_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \" \";\n")
          f.write("      if ((*(params->p" + name + "))->pKernelName == nullptr) {\n")
          f.write("        stream << \"0\";\n")
          f.write("      } else if (strlen((*(params->p" + name + "))->pKernelName) == 0) {\n")
          f.write("        stream << \" " + name + " = \\\"\\\"\";\n")
          f.write("      } else {\n")
          f.write("        stream << (*(params->p" + name + "))->pKernelName << \"}\";\n")
          f.write("      }\n")
          f.write("    }\n")
        elif type.find("ze_device_mem_alloc_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->ordinal << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_context_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_command_list_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->commandQueueGroupOrdinal << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_event_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->index << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->signal << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->wait << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_fence_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_image_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->type << \" \";\n")
          f.write("      stream << \"{\" << (*(params->p" + name +"))->format.layout << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->format.type << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->format.x << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->format.y << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->format.z << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->format.w << \"}\" << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->width << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->height << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->depth << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->arraylevels << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->miplevels << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_host_mem_alloc_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_external_memory_export_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_module_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->format << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->inputSize << \" \";\n")
          f.write("      stream << static_cast<const void*>((*(params->p" + name + "))->pInputModule) << \" \";\n")
          f.write("      if ((*(params->p" + name + ")) -> pBuildFlags != nullptr) \n")
          f.write("        stream << (*(params->p" + name + "))->pBuildFlags << \" \";\n")
          f.write("      else stream << 0 << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->pConstants << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_sampler_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->addressMode << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->filterMode << \" \";\n")
          f.write("      stream << static_cast<int>((*(params->p" + name + "))->isNormalized) << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_physical_mem_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->size << \"}\";\n")
          f.write("    }\n")
        elif type.find("ze_raytracing_mem_alloc_ext_desc_t*") >= 0:
          f.write("    if (*(params->p" + name + ") != nullptr) {\n")
          f.write("      stream << \" {\" << GetStructureTypeString((*(params->p" + name + "))->stype)\n")
          f.write("        << \"(0x\" << std::hex << (*(params->p" + name + "))->stype << std::dec << \") \";\n")
          f.write("      stream << (*(params->p" + name + "))->pNext << \" \";\n")
          f.write("      stream << (*(params->p" + name + "))->flags << \"}\";\n")
          f.write("    }\n")
  f.write("    stream << std::endl;\n")
  f.write("    collector->correlator_->Log(stream.str());\n")
  f.write("  }\n")

def gen_exit_callback(f, func, params, enum_map):
  f.write("  PTI_ASSERT(global_user_data != nullptr);\n")
  f.write("  ZeApiCollector* collector =\n")
  f.write("    reinterpret_cast<ZeApiCollector*>(global_user_data);\n")
  f.write("  uint64_t end_time = collector->GetTimestamp();\n")
  f.write("\n")
  f.write("  PTI_ASSERT(collector->correlator_ != nullptr);\n")
  f.write("\n")
  f.write("  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);\n")
  f.write("  PTI_ASSERT(start_time > 0);\n")
  f.write("  PTI_ASSERT(start_time < end_time);\n")
  f.write("  uint64_t time = end_time - start_time;\n")
  f.write("  collector->AddFunctionTime(\"" + func + "\", time);\n")
  f.write("  if (collector->options_.call_tracing) {\n")
  f.write("    std::stringstream stream;\n")
  f.write("    stream << \"<<<< [\" << end_time << \"] \";\n")
  f.write("    if (collector->options_.need_pid) {\n")
  f.write("      stream << \"<PID:\" << utils::GetPid() << \"> \";\n")
  f.write("    }\n")
  f.write("    if (collector->options_.need_tid) {\n")
  f.write("      stream << \"<TID:\" << utils::GetTid() << \"> \";\n")
  f.write("    }\n")
  f.write("    stream << \"" + func + "\";\n")
  if func == "zeCommandListAppendLaunchKernel" or\
     func == "zeCommandListAppendLaunchCooperativeKernel" or\
     func == "zeCommandListAppendLaunchKernelIndirect" or\
     func == "zeCommandListAppendMemoryCopy" or\
     func == "zeCommandListAppendMemoryFill" or\
     func == "zeCommandListAppendBarrier" or\
     func == "zeCommandListAppendMemoryRangesBarrier" or\
     func == "zeCommandListAppendMemoryCopyRegion" or\
     func == "zeCommandListAppendMemoryCopyFromContext" or\
     func == "zeCommandListAppendImageCopy" or\
     func == "zeCommandListAppendImageCopyRegion" or\
     func == "zeCommandListAppendImageCopyToMemory" or\
     func == "zeCommandListAppendImageCopyFromMemory":
    f.write("    uint64_t kernel_id = collector->correlator_->GetKernelId();\n")
    f.write("    if (kernel_id > 0) {\n")
    f.write("      stream << \"(\" << kernel_id << \")\";\n")
    f.write("    }\n")
  elif func == "zeCommandQueueExecuteCommandLists":
    f.write("    uint32_t command_list_count = *(params->pnumCommandLists);\n")
    f.write("    ze_command_list_handle_t* command_lists = *(params->pphCommandLists);\n")
    f.write("    std::string kernel_call_id;\n")
    f.write("    if (command_lists != nullptr) {\n")
    f.write("      for (uint32_t i = 0; i < command_list_count; ++i) {\n")
    f.write("        std::vector<uint64_t> kernel_id_list =\n")
    f.write("          collector->correlator_->GetKernelId(command_lists[i]);\n")
    f.write("        std::vector<uint64_t> call_id_list =\n")
    f.write("          collector->correlator_->GetCallId(command_lists[i]);\n")
    f.write("        PTI_ASSERT(kernel_id_list.size() == call_id_list.size());\n")
    f.write("        for (size_t j = 0; j < kernel_id_list.size(); ++j) {\n")
    f.write("          kernel_call_id += std::to_string(kernel_id_list[j]) + \".\"\n")
    f.write("            + std::to_string(call_id_list[j]) + \",\";\n")
    f.write("        }\n")
    f.write("      }\n")
    f.write("    }\n")
    f.write("    \n")
    f.write("    if (!kernel_call_id.empty()) {\n")
    f.write("      kernel_call_id = kernel_call_id.substr(0, kernel_call_id.size() - 1);\n")
    f.write("      stream << \"(\" << kernel_call_id << \")\";\n")
    f.write("    }\n")
  f.write("    stream << \" [\" << time << \" ns]\";\n")
  for name, type in params:
    if name.find("ph") == 0 or name.find("pptr") == 0 or name.find("pCount") == 0:
      f.write("    if (*(params->p" + name + ") != nullptr) {\n")
      if type == "ze_ipc_mem_handle_t*" or type == "ze_ipc_event_pool_handle_t*":
        f.write("      stream << \" " + name[1:] + " = \" << (*(params->p" + name + "))->data;\n")
      else:
        f.write("      stream << \" " + name[1:] + " = \" << **(params->p" + name + ");\n")
      f.write("    }\n")
    elif name.find("groupSize") == 0 and type.find("uint32_t*") == 0:
      f.write("    if (*(params->p" + name + ") != nullptr) {\n")
      f.write("      stream << \" " + name + " = \" << **(params->p" + name + ");\n")
      f.write("    }\n")
  f.write("    stream << \" -> \" << GetResultString(result) << \n")
  f.write("      \"(0x\" << result << \")\" << std::endl;\n")
  f.write("    collector->correlator_->Log(stream.str());\n")
  f.write("  }\n")
  f.write("\n")
  f.write("  if (collector->callback_ != nullptr) {\n")
  if func == "zeCommandListAppendLaunchKernel" or\
     func == "zeCommandListAppendLaunchCooperativeKernel" or\
     func == "zeCommandListAppendLaunchKernelIndirect" or\
     func == "zeCommandListAppendMemoryCopy" or\
     func == "zeCommandListAppendMemoryFill" or\
     func == "zeCommandListAppendBarrier" or\
     func == "zeCommandListAppendMemoryRangesBarrier" or\
     func == "zeCommandListAppendMemoryCopyRegion" or\
     func == "zeCommandListAppendMemoryCopyFromContext" or\
     func == "zeCommandListAppendImageCopy" or\
     func == "zeCommandListAppendImageCopyRegion" or\
     func == "zeCommandListAppendImageCopyToMemory" or\
     func == "zeCommandListAppendImageCopyFromMemory":
    f.write("    collector->callback_(\n")
    f.write("        collector->callback_data_,\n")
    f.write("        std::to_string(collector->correlator_->GetKernelId()),\n")
    f.write("        \"" + func + "\",\n")
    f.write("        start_time, end_time);\n")
  elif func == "zeCommandQueueExecuteCommandLists":
    f.write("    uint32_t command_list_count = *(params->pnumCommandLists);\n")
    f.write("    ze_command_list_handle_t* command_lists = *(params->pphCommandLists);\n")
    f.write("    std::string kernel_call_id;\n")
    f.write("    if (command_lists != nullptr) {\n")
    f.write("      for (uint32_t i = 0; i < command_list_count; ++i) {\n")
    f.write("        std::vector<uint64_t> kernel_id_list =\n")
    f.write("          collector->correlator_->GetKernelId(command_lists[i]);\n")
    f.write("        std::vector<uint64_t> call_id_list =\n")
    f.write("          collector->correlator_->GetCallId(command_lists[i]);\n")
    f.write("        PTI_ASSERT(kernel_id_list.size() == call_id_list.size());\n")
    f.write("        for (size_t j = 0; j < kernel_id_list.size(); ++j) {\n")
    f.write("          kernel_call_id += std::to_string(kernel_id_list[j]) + \".\"\n")
    f.write("            + std::to_string(call_id_list[j]) + \",\";\n")
    f.write("        }\n")
    f.write("      }\n")
    f.write("    }\n")
    f.write("    \n")
    f.write("    if (!kernel_call_id.empty()) {\n")
    f.write("      kernel_call_id = kernel_call_id.substr(0, kernel_call_id.size() - 1);\n")
    f.write("    } else {\n")
    f.write("      kernel_call_id = \"0\";\n")
    f.write("    }\n")
    f.write("    collector->callback_(\n")
    f.write("        collector->callback_data_, kernel_call_id,\n")
    f.write("        \"" + func + "\",\n")
    f.write("        start_time, end_time);\n")
  else:
    f.write("    collector->callback_(\n")
    f.write("        collector->callback_data_, \"0\",\n")
    f.write("        \"" + func + "\",\n")
    f.write("        start_time, end_time);\n")
  f.write("  }\n")

def gen_callbacks(f, func_list, group_map, param_map, enum_map):
  for func in func_list:
    assert func in group_map
    assert func in param_map
    group, callback = group_map[func]
    callback_name = callback[0]
    callback_cond = callback[1]
    if callback_cond:
      f.write("#if " + callback_cond + "\n")
    f.write("static void " + func + "OnEnter(\n")
    f.write("    " + get_param_struct_name(func) + "* params,\n")
    f.write("    ze_result_t result,\n")
    f.write("    void* global_user_data,\n")
    f.write("    void** instance_user_data) {\n")
    gen_enter_callback(f, func, param_map[func], enum_map)
    f.write("}\n")
    f.write("\n")
    f.write("static void " + func + "OnExit(\n")
    f.write("    " + get_param_struct_name(func) + "* params,\n")
    f.write("    ze_result_t result,\n")
    f.write("    void* global_user_data,\n")
    f.write("    void** instance_user_data) {\n")
    gen_exit_callback(f, func, param_map[func], enum_map)
    f.write("}\n")
    if callback_cond:
      f.write("#endif //" + callback_cond + "\n")
    f.write("\n")

def main():
  if len(sys.argv) < 3:
    print("Usage: python gen_tracing_header.py <output_include_path> <l0_include_path>")
    return

  dst_path = sys.argv[1]
  if (not os.path.exists(dst_path)):
    os.mkdir(dst_path)

  dst_file_path = os.path.join(dst_path, "tracing.gen")
  if (os.path.isfile(dst_file_path)):
    os.remove(dst_file_path)

  dst_file = open(dst_file_path, "wt")

  l0_path = sys.argv[2]
  l0_file_path = os.path.join(l0_path, "ze_api.h")

  l0_file = open(l0_file_path, "rt")
  func_list = get_func_list(l0_file)
  group_map = get_callback_group_map(l0_file)
  param_map = get_param_map(l0_file)
  enum_map = get_enum_map(l0_path)

  gen_result_converter(dst_file, enum_map)
  gen_structure_type_converter(dst_file, enum_map)
  gen_callbacks(dst_file, func_list, group_map, param_map, enum_map)
  gen_api(dst_file, func_list, group_map)

  l0_file.close()
  dst_file.close()

if __name__ == "__main__":
  main()