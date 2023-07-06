package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hlo_evaluator",
    srcs = [
        "hlo_evaluator.cc",
        "hlo_evaluator_typed_visitor.h",
        "hlo_evaluator_typed_visitor_bfloat16.cc",
        "hlo_evaluator_typed_visitor_bool.cc",
        "hlo_evaluator_typed_visitor_complex128.cc",
        "hlo_evaluator_typed_visitor_complex64.cc",
        "hlo_evaluator_typed_visitor_double.cc",
        "hlo_evaluator_typed_visitor_float.cc",
        "hlo_evaluator_typed_visitor_half.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        ":dynamic_dimension_inference",
        ":hlo",
        ":hlo_query",
        ":pattern_matcher",
        ":shape_inference",
        "//itex/core/compiler/xla:array2d",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/utils:common_utils",
        "//protos:xla_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_live_range",
    srcs = [
        "hlo_live_range.cc",
    ],
    hdrs = [
        "hlo_live_range.h",
    ],
    deps = [
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo",
    srcs = [
        "dfs_hlo_visitor.cc",
        "dynamic_parameter_binding.cc",
        "hlo_computation.cc",
        "hlo_input_output_alias_config.cc",
        "hlo_instruction.cc",
        "hlo_instructions.cc",
        "hlo_module.cc",
        "hlo_module_metadata.cc",
        "hlo_op_metadata.cc",
        "hlo_opcode.cc",
        "hlo_schedule.cc",
        "hlo_sharding.cc",
        "hlo_sharding_metadata.cc",
    ],
    hdrs = [
        "dfs_hlo_visitor.h",
        "dfs_hlo_visitor_with_default.h",
        "dynamic_parameter_binding.h",
        "hlo_casting_utils.h",
        "hlo_clone_context.h",
        "hlo_computation.h",
        "hlo_domain_metadata.h",
        "hlo_input_output_alias_config.h",
        "hlo_instruction.h",
        "hlo_instructions.h",
        "hlo_module.h",
        "hlo_module_metadata.h",
        "hlo_op_metadata.h",
        "hlo_opcode.h",
        "hlo_schedule.h",
        "hlo_sharding.h",
        "hlo_sharding_metadata.h",
    ],
    deps = [
        ":hlo_module_config",
        ":mapped_ptr_container_sorter",
        ":name_uniquer",
        "//itex/core/compiler/xla:array",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:protobuf_util",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    deps = [
        ":buffer_assignment",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_proto_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pattern_matcher",
    hdrs = ["pattern_matcher.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
    ],
)

cc_library(
    name = "pattern_matcher_gmock",
    testonly = 1,
    hdrs = ["pattern_matcher_gmock.h"],
    deps = [
        ":pattern_matcher",
        "//itex/core/compiler/xla:test",
        "//itex/core/utils:test",
    ],
)

cc_library(
    name = "hlo_reachability",
    srcs = ["hlo_reachability.cc"],
    hdrs = ["hlo_reachability.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "call_graph",
    srcs = ["call_graph.cc"],
    hdrs = ["call_graph.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo_graph_dumper",
    srcs = ["hlo_graph_dumper.cc"],
    hdrs = ["hlo_graph_dumper.h"],
    deps = [
        ":hlo",
        ":hlo_execution_profile",
        ":pattern_matcher",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_execution_profile",
    srcs = ["hlo_execution_profile.cc"],
    hdrs = ["hlo_execution_profile.h"],
    deps = [
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_profile_printer",
        ":human_readable_profile_builder",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_cost_analysis",
    srcs = ["hlo_cost_analysis.cc"],
    hdrs = ["hlo_cost_analysis.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_profile_printer",
    srcs = ["hlo_profile_printer.cc"],
    hdrs = ["hlo_profile_printer.h"],
    deps = [
        ":human_readable_profile_builder",
        "//itex/core/compiler/xla:types",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "human_readable_profile_builder",
    srcs = ["human_readable_profile_builder.cc"],
    hdrs = ["human_readable_profile_builder.h"],
    deps = [
        "//itex/core/compiler/xla:metric_table_report",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "fusion_node_indexing_evaluation",
    srcs = ["fusion_node_indexing_evaluation.cc"],
    hdrs = ["fusion_node_indexing_evaluation.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "name_uniquer",
    srcs = ["name_uniquer.cc"],
    hdrs = ["name_uniquer.h"],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "buffer_assignment",
    srcs = [
        "buffer_assignment.cc",
    ],
    hdrs = [
        "buffer_assignment.h",
    ],
    deps = [
        ":buffer_value_containers",
        ":heap_simulator",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_live_range",
        ":hlo_value",
        ":logical_buffer",
        ":memory_space_assignment",
        ":tuple_points_to_analysis",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_ordering",
    srcs = ["hlo_ordering.cc"],
    hdrs = ["hlo_ordering.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_reachability",
        ":hlo_value",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo_runner_interface",
    srcs = ["hlo_runner_interface.cc"],
    hdrs = ["hlo_runner_interface.h"],
    deps = [
        ":computation_placer",
        ":executable",
        ":hlo",
        ":hlo_parser",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "heap_simulator",
    srcs = ["heap_simulator.cc"],
    hdrs = ["heap_simulator.h"],
    deps = [
        ":buffer_value",
        ":buffer_value_containers",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_live_range",
        ":hlo_ordering",
        ":memory_space_assignment_repacking",
        ":tuple_points_to_analysis",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_matchers",
    testonly = 1,
    srcs = ["hlo_matchers.cc"],
    hdrs = ["hlo_matchers.h"],
    deps = [
        ":hlo",
        ":hlo_parser",
        "//itex/core/compiler/xla:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "hlo_module_config",
    srcs = ["hlo_module_config.cc"],
    hdrs = ["hlo_module_config.h"],
    deps = [
        ":computation_layout",
        ":computation_placer",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:shape_layout",
        "//itex/core/compiler/xla:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "computation_layout",
    srcs = ["computation_layout.cc"],
    hdrs = ["computation_layout.h"],
    deps = [
        "//itex/core/compiler/xla:shape_layout",
        "//itex/core/compiler/xla:types",
        "//protos:tf_protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "computation_placer",
    srcs = ["computation_placer.cc"],
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "//itex/core/compiler/xla:array2d",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:platform",
        "//itex/core/compiler/xla/stream_executor/host:host_platform_id",
        "//itex/core/compiler/xla/stream_executor/sycl:sycl_platform_id",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "global_device_id",
    srcs = ["global_device_id.cc"],
    hdrs = ["global_device_id.h"],
    deps = [
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "maybe_owning_device_memory",
    srcs = [
        "maybe_owning_device_memory.cc",
    ],
    hdrs = [
        "maybe_owning_device_memory.h",
    ],
    deps = [
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/compiler/xla/service/llvm_ir:ir_array",
        "//itex/core/compiler/xla/service/llvm_ir:ir_builder_mixin",
        "//itex/core/compiler/xla/service/llvm_ir:llvm_loop",
        "//itex/core/compiler/xla/service/llvm_ir:llvm_util",
        "//itex/core/compiler/xla/service/llvm_ir:loop_emitter",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)

cc_library(
    name = "buffer_value",
    srcs = ["buffer_value.cc"],
    hdrs = ["buffer_value.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "buffer_value_containers",
    hdrs = ["buffer_value_containers.h"],
    deps = [
        ":buffer_value",
        ":logical_buffer",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "logical_buffer",
    srcs = ["logical_buffer.cc"],
    hdrs = ["logical_buffer.h"],
    deps = [
        ":buffer_value",
        ":hlo",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_value",
    srcs = ["hlo_value.cc"],
    hdrs = ["hlo_value.h"],
    deps = [
        ":buffer_value",
        ":hlo",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_dataflow_analysis",
    srcs = ["hlo_dataflow_analysis.cc"],
    hdrs = ["hlo_dataflow_analysis.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_phi_graph",
        ":hlo_value",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_phi_graph",
    srcs = ["hlo_phi_graph.cc"],
    hdrs = ["hlo_phi_graph.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_value",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_buffer",
    srcs = ["hlo_buffer.cc"],
    hdrs = ["hlo_buffer.h"],
    deps = [
        ":hlo",
        ":hlo_value",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_alias_analysis",
    srcs = ["hlo_alias_analysis.cc"],
    hdrs = ["hlo_alias_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "logical_buffer_analysis",
    srcs = ["logical_buffer_analysis.cc"],
    hdrs = ["logical_buffer_analysis.h"],
    deps = [
        ":hlo",
        ":logical_buffer",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tuple_points_to_analysis",
    srcs = ["tuple_points_to_analysis.cc"],
    hdrs = ["tuple_points_to_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":logical_buffer",
        ":logical_buffer_analysis",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "memory_space_assignment_utils",
    srcs = ["memory_space_assignment_utils.cc"],
    hdrs = ["memory_space_assignment_utils.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
    ],
)

cc_library(
    name = "memory_space_assignment_tuning_utils",
    srcs = ["memory_space_assignment_tuning_utils.cc"],
    hdrs = ["memory_space_assignment_tuning_utils.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_module_config",
        ":memory_space_assignment_utils",
    ],
)

cc_library(
    name = "memory_space_assignment_repacking",
    hdrs = ["memory_space_assignment_repacking.h"],
    deps = [
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
    ],
)

cc_library(
    name = "memory_space_assignment",
    srcs = ["memory_space_assignment.cc"],
    hdrs = ["memory_space_assignment.h"],
    deps = [
        ":heap_simulator",
        ":hlo_cost_analysis",
        ":memory_space_assignment_repacking",
        ":memory_space_assignment_tuning_utils",
        ":memory_space_assignment_utils",
        ":tuple_util",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
    ],
)

cc_library(
    name = "tuple_util",
    srcs = ["tuple_util.cc"],
    hdrs = ["tuple_util.h"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_element_type_converter",
    srcs = ["hlo_element_type_converter.cc"],
    hdrs = ["hlo_element_type_converter.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_pass",
        ":hlo_query",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "hlo_module_group",
    srcs = ["hlo_module_group.cc"],
    hdrs = ["hlo_module_group.h"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_module_util",
    srcs = ["hlo_module_util.cc"],
    hdrs = ["hlo_module_util.h"],
    deps = [
        ":compiler",
        ":hlo",
        ":hlo_module_config",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_memory_scheduler",
    srcs = ["hlo_memory_scheduler.cc"],
    hdrs = ["hlo_memory_scheduler.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "hlo_pass",
    hdrs = [
        "hlo_pass_fix.h",
        "hlo_pass_interface.h",
    ],
    deps = [
        ":hlo",
        ":hlo_module_group",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "hlo_pass_pipeline",
    srcs = [
        "hlo_pass_pipeline.cc",
    ],
    hdrs = [
        "hlo_pass_pipeline.h",
    ],
    deps = [
        ":compilation_stats",
        ":dump",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_proto_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo_sharding_util",
    srcs = [
        "hlo_sharding_util.cc",
    ],
    hdrs = [
        "hlo_sharding_util.h",
    ],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:array",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "sharding_propagation",
    srcs = [
        "sharding_propagation.cc",
    ],
    hdrs = [
        "sharding_propagation.h",
    ],
    deps = [
        ":dot_as_convolution_util",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_sharding_util",
        "//itex/core/compiler/xla:protobuf_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:sharding_op_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sharding_remover",
    srcs = [
        "sharding_remover.cc",
    ],
    hdrs = [
        "sharding_remover.h",
    ],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "dot_as_convolution_util",
    srcs = [
        "dot_as_convolution_util.cc",
    ],
    hdrs = [
        "dot_as_convolution_util.h",
    ],
    deps = [
        ":hlo",
        ":shape_inference",
        "//itex/core/compiler/xla:status_macros",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "stream_pool",
    srcs = ["stream_pool.cc"],
    hdrs = ["stream_pool.h"],
    deps = [
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/stream_executor/sycl:sycl_executor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_proto_util",
    srcs = ["hlo_proto_util.cc"],
    hdrs = ["hlo_proto_util.h"],
    deps = [
        ":buffer_assignment",
        ":hlo",
        ":hlo_verifier",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:util",
    ],
)

cc_library(
    name = "hlo_verifier",
    srcs = ["hlo_verifier.cc"],
    hdrs = ["hlo_verifier.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo",
        ":hlo_pass",
        ":shape_inference",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_parser",
    srcs = ["hlo_parser.cc"],
    hdrs = ["hlo_parser.h"],
    deps = [
        ":hlo",
        ":hlo_lexer",
        ":shape_inference",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = [
        "hlo_lexer.h",
    ],
    deps = [
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "shaped_buffer",
    srcs = ["shaped_buffer.cc"],
    hdrs = ["shaped_buffer.h"],
    deps = [
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor:device_memory",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = [
        "executable.h",
        "service_executable_run_options.h",
    ],
    deps = [
        "computation_placer",
        ":computation_layout",
        ":dump",
        ":hlo",
        ":hlo_execution_profile",
        ":hlo_graph_dumper",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        ":stream_pool",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "xla_debug_info_manager",
    srcs = [
        "xla_debug_info_manager.cc",
    ],
    hdrs = [
        "xla_debug_info_manager.h",
    ],
    deps = [
        ":hlo",
        ":hlo_proto_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "hlo_creation_utils",
    srcs = ["hlo_creation_utils.cc"],
    hdrs = [
        "hlo_creation_utils.h",
    ],
    deps = [
        ":hlo",
        ":hlo_module_config",
        ":shape_inference",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/client/lib:comparators",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compilation_stats",
    srcs = ["compilation_stats.cc"],
    hdrs = ["compilation_stats.h"],
    deps = [
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "algebraic_simplifier",
    srcs = ["algebraic_simplifier.cc"],
    hdrs = ["algebraic_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_evaluator",
        ":hlo_pass",
        ":hlo_query",
        ":pattern_matcher",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_comparison",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "all_gather_combiner",
    srcs = ["all_gather_combiner.cc"],
    hdrs = ["all_gather_combiner.h"],
    deps = [
        ":collective_combiner_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//itex/core/compiler/xla:array2d",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "all_gather_broadcast_reorder",
    srcs = ["all_gather_broadcast_reorder.cc"],
    hdrs = ["all_gather_broadcast_reorder.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "all_reduce_combiner",
    srcs = ["all_reduce_combiner.cc"],
    hdrs = ["all_reduce_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//itex/core/compiler/xla:array2d",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "all_reduce_contiguous",
    srcs = ["all_reduce_contiguous.cc"],
    hdrs = ["all_reduce_contiguous.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
    ],
)

cc_library(
    name = "all_reduce_key",
    srcs = ["all_reduce_key.cc"],
    hdrs = ["all_reduce_key.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        "//itex/core/compiler/xla:statusor",
    ],
)

cc_library(
    name = "all_reduce_reassociate",
    srcs = ["all_reduce_reassociate.cc"],
    hdrs = ["all_reduce_reassociate.h"],
    deps = [
        ":all_reduce_key",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "all_reduce_folder",
    srcs = ["all_reduce_folder.cc"],
    hdrs = ["all_reduce_folder.h"],
    deps = [
        ":all_reduce_key",
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "all_to_all_decomposer",
    srcs = ["all_to_all_decomposer.cc"],
    hdrs = ["all_to_all_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "async_collective_creator",
    srcs = ["async_collective_creator.cc"],
    hdrs = ["async_collective_creator.h"],
    deps = [
        # ":computation_placer_hdr",
        ":hlo",
        ":hlo_pass",
        ":shape_inference",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "batchnorm_expander",
    srcs = ["batchnorm_expander.cc"],
    hdrs = ["batchnorm_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bfloat16_support",
    srcs = ["bfloat16_support.cc"],
    hdrs = ["bfloat16_support.h"],
    deps = [
        ":hlo",
    ],
)

cc_library(
    name = "bfloat16_normalization",
    srcs = ["bfloat16_normalization.cc"],
    hdrs = ["bfloat16_normalization.h"],
    deps = [
        ":bfloat16_support",
        ":hlo",
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_simplifier",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bitcast_dtypes_expander",
    srcs = ["bitcast_dtypes_expander.cc"],
    hdrs = ["bitcast_dtypes_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client/lib:arithmetic",
        "//itex/core/compiler/xla/client/lib:broadcast",
        "//itex/core/compiler/xla/client/lib:constants",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "collective_combiner_utils",
    hdrs = ["collective_combiner_utils.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_reachability",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "collectives_schedule_linearizer",
    srcs = ["collectives_schedule_linearizer.cc"],
    hdrs = ["collectives_schedule_linearizer.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//itex/core/compiler/xla:array2d",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_4d_expander",
    srcs = ["convolution_4d_expander.cc"],
    hdrs = ["convolution_4d_expander.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convert_mover",
    srcs = ["convert_mover.cc"],
    hdrs = ["convert_mover.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//itex/core/compiler/xla/service/graphcycles",
    ],
)

cc_library(
    name = "flatten_call_graph",
    srcs = ["flatten_call_graph.cc"],
    hdrs = ["flatten_call_graph.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "optimization_barrier_expander",
    srcs = ["optimization_barrier_expander.cc"],
    hdrs = ["optimization_barrier_expander.h"],
    deps = [
        ":op_expander_pass",
    ],
)

cc_library(
    name = "call_inliner",
    srcs = ["call_inliner.cc"],
    hdrs = ["call_inliner.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_dce",
        ":hlo_domain_isolator",
        ":hlo_pass",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "comparison_expander",
    srcs = ["comparison_expander.cc"],
    hdrs = ["comparison_expander.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client/lib:comparators",
    ],
)

cc_library(
    name = "conditional_canonicalizer",
    srcs = ["conditional_canonicalizer.cc"],
    hdrs = ["conditional_canonicalizer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:status_macros",
    ],
)

cc_library(
    name = "conditional_simplifier",
    srcs = ["conditional_simplifier.cc"],
    hdrs = ["conditional_simplifier.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_call_target_registry",
    srcs = ["custom_call_target_registry.cc"],
    hdrs = ["custom_call_target_registry.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "custom_call_status",
    srcs = [
        "custom_call_status.cc",
        "custom_call_status_internal.h",
    ],
    hdrs = ["custom_call_status.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "custom_call_status_internal",
    hdrs = [
        "custom_call_status_internal.h",
    ],
    visibility = [
        ":__subpackages__",
    ],
    deps = [
        ":custom_call_status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["dot_decomposer.cc"],
    hdrs = ["dot_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dot_merger",
    srcs = ["dot_merger.cc"],
    hdrs = ["dot_merger.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":shape_inference",
        "//itex/core/compiler/xla/service/graphcycles",
    ],
)

cc_library(
    name = "dynamic_dimension_simplifier",
    srcs = ["dynamic_dimension_simplifier.cc"],
    hdrs = ["dynamic_dimension_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:status_macros",
    ],
)

cc_library(
    name = "dynamic_index_splitter",
    srcs = ["dynamic_index_splitter.cc"],
    hdrs = ["dynamic_index_splitter.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dynamic_padder",
    srcs = ["dynamic_padder.cc"],
    hdrs = ["dynamic_padder.h"],
    deps = [
        ":dynamic_dimension_inference",
        ":dynamic_window_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_dce",
        ":hlo_pass",
        ":pattern_matcher",
        ":shape_inference",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "eigh_expander",
    srcs = ["eigh_expander.cc"],
    hdrs = ["eigh_expander.h"],
    deps = [
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client/lib:arithmetic",
        "//itex/core/compiler/xla/client/lib:comparators",
        "//itex/core/compiler/xla/client/lib:constants",
        "//itex/core/compiler/xla/client/lib:loops",
        "//itex/core/compiler/xla/client/lib:math",
        "//itex/core/compiler/xla/client/lib:matrix",
        "//itex/core/compiler/xla/client/lib:slicing",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "gather_scatter_utils",
    srcs = ["gather_scatter_utils.cc"],
    hdrs = ["gather_scatter_utils.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        "//itex/core/compiler/xla:permutation_util",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gather_simplifier",
    srcs = ["gather_simplifier.cc"],
    hdrs = ["gather_simplifier.h"],
    deps = [
        ":gather_scatter_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "hlo_cse",
    srcs = ["hlo_cse.cc"],
    hdrs = ["hlo_cse.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "hlo_dce",
    srcs = ["hlo_dce.cc"],
    hdrs = ["hlo_dce.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "hlo_constant_folding",
    srcs = ["hlo_constant_folding.cc"],
    hdrs = ["hlo_constant_folding.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_pass",
        ":hlo_query",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_domain_map",
    srcs = ["hlo_domain_map.cc"],
    hdrs = ["hlo_domain_map.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_domain_verifier",
    srcs = ["hlo_domain_verifier.cc"],
    hdrs = ["hlo_domain_verifier.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "hlo_domain_isolator",
    srcs = ["hlo_domain_isolator.cc"],
    hdrs = ["hlo_domain_isolator.h"],
    deps = [
        ":hlo",
        ":hlo_domain_remover",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        # "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "hlo_domain_remover",
    srcs = ["hlo_domain_remover.cc"],
    hdrs = ["hlo_domain_remover.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_domain_verifier",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//itex/core/compiler/xla:types",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "hlo_query",
    srcs = ["hlo_query.cc"],
    hdrs = ["hlo_query.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "fusion_queue",
    hdrs = ["fusion_queue.h"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gather_expander",
    srcs = ["gather_expander.cc"],
    hdrs = ["gather_expander.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        ":while_util",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":fusion_queue",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_reachability",
        ":pattern_matcher",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "logistic_expander",
    srcs = ["logistic_expander.cc"],
    hdrs = ["logistic_expander.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "copy_insertion",
    srcs = ["copy_insertion.cc"],
    hdrs = [
        "compile_time_cap.h",
        "copy_insertion.h",
    ],
    deps = [
        ":dump",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_simplifier",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "layout_assignment",
    srcs = [
        "layout_assignment.cc",
    ],
    hdrs = [
        "layout_assignment.h",
    ],
    deps = [
        ":call_graph",
        ":computation_layout",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        ":tuple_simplifier",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_layout",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "loop_schedule_linearizer",
    srcs = ["loop_schedule_linearizer.cc"],
    hdrs = ["loop_schedule_linearizer.h"],
    deps = [
        ":dump",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_simplifier",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service/graphcycles",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "operand_upcaster",
    srcs = ["operand_upcaster.cc"],
    hdrs = ["operand_upcaster.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        ":shape_inference",
    ],
)

cc_library(
    name = "op_expander_pass",
    srcs = ["op_expander_pass.cc"],
    hdrs = ["op_expander_pass.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "qr_expander",
    srcs = ["qr_expander.cc"],
    hdrs = ["qr_expander.h"],
    deps = [
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client/lib:arithmetic",
        "//itex/core/compiler/xla/client/lib:constants",
        "//itex/core/compiler/xla/client/lib:loops",
        "//itex/core/compiler/xla/client/lib:math",
        "//itex/core/compiler/xla/client/lib:matrix",
        "//itex/core/compiler/xla/client/lib:qr",
        "//itex/core/compiler/xla/client/lib:slicing",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "real_imag_expander",
    srcs = ["real_imag_expander.cc"],
    hdrs = ["real_imag_expander.h"],
    deps = [
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
    ],
)

cc_library(
    name = "reduce_scatter_utils",
    srcs = ["reduce_scatter_utils.cc"],
    hdrs = ["reduce_scatter_utils.h"],
    deps = [
        ":hlo",
    ],
)

cc_library(
    name = "reduce_scatter_combiner",
    srcs = ["reduce_scatter_combiner.cc"],
    hdrs = ["reduce_scatter_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//itex/core/compiler/xla:array2d",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reshape_mover",
    srcs = ["reshape_mover.cc"],
    hdrs = ["reshape_mover.h"],
    deps = [
        ":hlo_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "result_caster",
    srcs = ["result_caster.cc"],
    hdrs = ["result_caster.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        ":shape_inference",
    ],
)

cc_library(
    name = "rng_bit_generator_expander",
    srcs = ["rng_bit_generator_expander.cc"],
    hdrs = ["rng_bit_generator_expander.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client/lib:prng",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "rng_expander",
    srcs = ["rng_expander.cc"],
    hdrs = ["rng_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla/client:xla_builder",
        "//itex/core/compiler/xla/client/lib:prng",
    ],
)

cc_library(
    name = "sort_simplifier",
    srcs = ["sort_simplifier.cc"],
    hdrs = ["sort_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "scatter_expander",
    srcs = ["scatter_expander.cc"],
    hdrs = ["scatter_expander.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_creation_utils",
        ":op_expander_pass",
        ":while_util",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "scatter_simplifier",
    srcs = ["scatter_simplifier.cc"],
    hdrs = ["scatter_simplifier.h"],
    deps = [
        ":gather_scatter_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//itex/core/compiler/xla:permutation_util",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "simplify_fp_conversions",
    srcs = ["simplify_fp_conversions.cc"],
    hdrs = ["simplify_fp_conversions.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:comparison_util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "stable_sort_expander",
    srcs = ["stable_sort_expander.cc"],
    hdrs = ["stable_sort_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":op_expander_pass",
        "//itex/core/compiler/xla:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "transpose_folding",
    srcs = ["transpose_folding.cc"],
    hdrs = ["transpose_folding.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "tuple_simplifier",
    srcs = ["tuple_simplifier.cc"],
    hdrs = ["tuple_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "while_loop_analysis",
    srcs = ["while_loop_analysis.cc"],
    hdrs = ["while_loop_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_reachability",
        ":pattern_matcher",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "while_loop_constant_sinking",
    srcs = ["while_loop_constant_sinking.cc"],
    hdrs = ["while_loop_constant_sinking.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":while_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "while_loop_simplifier",
    srcs = ["while_loop_simplifier.cc"],
    hdrs = ["while_loop_simplifier.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        ":pattern_matcher",
        ":while_loop_analysis",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:union_find",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "while_loop_trip_count_annotator",
    srcs = ["while_loop_trip_count_annotator.cc"],
    hdrs = ["while_loop_trip_count_annotator.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":while_loop_analysis",
        "//itex/core/compiler/xla:statusor",
    ],
)

cc_library(
    name = "zero_sized_hlo_elimination",
    srcs = ["zero_sized_hlo_elimination.cc"],
    hdrs = ["zero_sized_hlo_elimination.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "collective_ops_utils",
    srcs = ["collective_ops_utils.cc"],
    hdrs = ["collective_ops_utils.h"],
    deps = [
        ":computation_placer",
        ":global_device_id",
        ":hlo",
        ":pattern_matcher",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "dynamic_window_utils",
    srcs = ["dynamic_window_utils.cc"],
    hdrs = ["dynamic_window_utils.h"],
    deps = [
        ":hlo",
        ":shape_inference",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
    ],
)

cc_library(
    name = "dynamic_dimension_inference",
    srcs = ["dynamic_dimension_inference.cc"],
    hdrs = ["dynamic_dimension_inference.h"],
    deps = [
        ":dynamic_window_utils",
        ":hlo",
        ":tuple_util",
        ":while_util",
        "//itex/core/compiler/xla:literal_util",
        "//itex/core/compiler/xla:shape_tree",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla:window_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "while_util",
    srcs = ["while_util.cc"],
    hdrs = ["while_util.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_creation_utils",
        ":tuple_util",
        "//itex/core/compiler/xla:literal_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":buffer_assignment",
        ":buffer_value",
        ":computation_placer",
        ":executable",
        ":hlo",
        ":hlo_module_config",
        ":hlo_module_group",
        ":logical_buffer",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llvm_compiler",
    srcs = ["llvm_compiler.cc"],
    hdrs = ["llvm_compiler.h"],
    deps = [
        ":compiler",
        "//itex/core/utils:common_utils",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "transfer_manager",
    srcs = ["transfer_manager.cc"],
    hdrs = ["transfer_manager.h"],
    deps = [
        ":compiler",
        ":executable",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service/gpu:gpu_device_info",
        "//itex/core/compiler/xla/service/gpu:launch_dimensions",
        "//itex/core/compiler/xla/service/gpu:stream_executor_util",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "generic_transfer_manager",
    srcs = ["generic_transfer_manager.cc"],
    hdrs = ["generic_transfer_manager.h"],
    deps = [
        ":transfer_manager",
        "//itex/core/compiler/xla:literal",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/service/gpu:gpu_device_info",
        "//itex/core/compiler/xla/service/gpu:launch_dimensions",
        "//itex/core/compiler/xla/service/gpu:stream_executor_util",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "mapped_ptr_container_sorter",
    hdrs = ["mapped_ptr_container_sorter.h"],
    deps = [
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":compiler",
        ":computation_placer",
        ":platform_util",
        ":stream_pool",
        ":transfer_manager",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/compiler/xla/stream_executor/host:host_platform_id",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "allocation_tracker",
    srcs = ["allocation_tracker.cc"],
    hdrs = ["allocation_tracker.h"],
    deps = [
        ":backend",
        ":transfer_manager",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "channel_tracker",
    srcs = ["channel_tracker.cc"],
    hdrs = ["channel_tracker.h"],
    deps = [
        ":hlo",
        "//itex/core/compiler/xla:status",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compilation_cache",
    srcs = ["compilation_cache.cc"],
    hdrs = ["compilation_cache.h"],
    deps = [
        ":executable",
        ":hlo_module_config",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "execution_tracker",
    srcs = ["execution_tracker.cc"],
    hdrs = ["execution_tracker.h"],
    deps = [
        ":backend",
        ":stream_pool",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "source_map_util",
    srcs = [],
    hdrs = ["source_map_util.h"],
    deps = [
        ":executable",
        "//itex/core/compiler/xla:status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "platform_util",
    srcs = ["platform_util.cc"],
    hdrs = ["platform_util.h"],
    deps = [
        ":compiler",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor/host:host_platform_id",
        "//itex/core/compiler/xla/stream_executor/sycl:sycl_platform_id",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":allocation_tracker",
        ":backend",
        ":channel_tracker",
        ":compilation_cache",
        ":compiler",
        ":computation_layout",
        ":computation_placer",
        ":dump",
        ":dynamic_dimension_inference",
        #":dynamic_padder",
        ":executable",
        ":execution_tracker",
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_evaluator",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_group",
        ":hlo_module_util",
        ":hlo_proto_util",
        ":platform_util",
        ":source_map_util",
        ":stream_pool",
        ":transfer_manager",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:executable_run_options",
        "//itex/core/compiler/xla:execution_options_util",
        "//itex/core/compiler/xla:service_interface",
        "//itex/core/compiler/xla:shape_layout",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library(
    name = "local_service",
    srcs = ["local_service.cc"],
    hdrs = ["local_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":executable",
        ":hlo",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_util",
        ":platform_util",
        ":service",
        ":shaped_buffer",
        "//itex/core/compiler/xla:execution_options_util",
        "//itex/core/compiler/xla:shape_layout",
        "//itex/core/compiler/xla:shape_util",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/compiler/xla/client:executable_build_options",
        "//itex/core/compiler/xla/client:xla_computation",
        "//itex/core/compiler/xla/stream_executor",
        "//itex/core/compiler/xla/stream_executor:device_memory_allocator",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compile_only_service",
    srcs = ["compile_only_service.cc"],
    hdrs = ["compile_only_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":dump",
        ":platform_util",
        ":service",
        "//itex/core/compiler/xla:debug_options_flags",
        "//itex/core/compiler/xla:status_macros",
        "//itex/core/compiler/xla:statusor",
        "//itex/core/compiler/xla:types",
        "//itex/core/compiler/xla:util",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
    ],
)
