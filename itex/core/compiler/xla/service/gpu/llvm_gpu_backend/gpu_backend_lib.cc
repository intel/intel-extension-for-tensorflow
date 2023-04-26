/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "itex/core/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "LLVMSPIRVLib.h"
#include "LLVMSPIRVOpts.h"
#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "itex/core/compiler/xla/service/dump.h"
#include "itex/core/compiler/xla/service/gpu/llvm_gpu_backend/dump_ir_pass.h"
#include "itex/core/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "itex/core/compiler/xla/service/llvm_ir/llvm_command_line_options.h"
#include "itex/core/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "itex/core/compiler/xla/service/llvm_ir/llvm_util.h"
#include "itex/core/compiler/xla/status_macros.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/env.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/path.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/SYCLLowerIR/LowerWGLocalMemory.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"

namespace itex_xla {
namespace gpu {
namespace {

static llvm::codegen::RegisterCodeGenFlags CGF;

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Convenience function for producing a name of a temporary compilation product
// from the input filename.
std::string MakeNameForTempProduct(absl::string_view input_filename,
                                   absl::string_view extension) {
  return ReplaceFilenameExtension(itex::io::Basename(input_filename),
                                  extension);
}

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPreparePass(*pass_registry);
}

// Returns the TargetMachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const HloModuleConfig& hlo_module_config, absl::string_view feature_str) {
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    ITEX_VLOG(2) << "Unable to find Target for triple '" << triple.str() << "'"
                 << " -- " << error;
    return nullptr;
  }

  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());

  // Set the verbose assembly options.
  target_options.MCOptions.AsmVerbose = false;

  // The selection of codegen optimization level is copied from function
  // GetCodeGenOptLevel in //third_party/llvm/llvm/tools/opt/opt.cpp.
  llvm::CodeGenOpt::Level codegen_opt_level;
  switch (hlo_module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = llvm::CodeGenOpt::Less;
      break;
    case 2:
      codegen_opt_level = llvm::CodeGenOpt::Default;
      break;
    case 3:
      codegen_opt_level = llvm::CodeGenOpt::Aggressive;
      break;
    default:
      codegen_opt_level = llvm::CodeGenOpt::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options,
      llvm::codegen::getExplicitRelocModel(),
      llvm::codegen::getExplicitCodeModel(), codegen_opt_level));
}

// Adds the standard LLVM optimization passes, based on the speed optimization
// level (opt_level) and size optimization level (size_level). Both module
// and function-level passes are added, so two pass managers are passed in and
// modified by this function.
void AddOptimizationPasses(unsigned opt_level, unsigned size_level,
                           llvm::TargetMachine* target_machine,
                           llvm::legacy::PassManagerBase* module_passes,
                           llvm::legacy::FunctionPassManager* function_passes,
                           int inline_threshold) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = opt_level;
  builder.SizeLevel = size_level;

  if (opt_level > 1) {
    builder.Inliner = llvm::createFunctionInliningPass(inline_threshold);
  } else {
    // Only inline functions marked with "alwaysinline".
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  // Disable loop unroll in LLVM
  builder.DisableUnrollLoops = 1;
  builder.LoopVectorize = 0;
  builder.SLPVectorize = 0;

  // NVPTX's early-as-possible passes include NVVM reflect.
  // if (target_machine)
  //   target_machine->adjustPassManager(builder);

  builder.populateFunctionPassManager(*function_passes);
  builder.populateModulePassManager(*module_passes);
}

// Refer to function `EmitAssemblyHelper::RunOptimizationPipeline` defined in
// clang/lib/CodeGen/BackendUtil.cpp.
void RunOptimizationPipeline(llvm::Module* module,
                             llvm::TargetMachine* target_machine) {
  llvm::Optional<llvm::PGOOptions> PGOOpt;
  llvm::PipelineTuningOptions PTO;
  PTO.LoopUnrolling = 1;
  PTO.LoopInterleaving = 1;
  PTO.LoopVectorization = 1;
  PTO.SLPVectorization = 1;
  PTO.MergeFunctions = 0;
  PTO.CallGraphProfile = 1;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassInstrumentationCallbacks PIC;
  llvm::PrintPassOptions PrintPassOpts;
  PrintPassOpts.Indent = 0;
  PrintPassOpts.SkipAnalyses = 0;
  llvm::StandardInstrumentations SI(module->getContext(), false, false,
                                    PrintPassOpts);
  SI.registerCallbacks(PIC, &FAM);
  llvm::PassBuilder PB(target_machine, PTO, PGOOpt, &PIC);

#define HANDLE_EXTENSION(Ext) \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"

  // Register the target library analysis directly and give it a customized
  // preset TLI.
  auto target_triple = llvm::Triple(module->getTargetTriple());
  auto TLII = std::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  FAM.registerPass([&] { return llvm::TargetLibraryAnalysis(*TLII); });

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM;

  llvm::OptimizationLevel Level = llvm::OptimizationLevel::O2;
  MPM = PB.buildPerModuleDefaultPipeline(Level);
  MPM.addPass(llvm::SYCLLowerWGLocalMemoryPass());
  MPM.addPass(llvm::VerifierPass());
  MPM.run(*module, MAM);
}

// LLVM has an extensive flags mechanism of its own, which is only accessible
// through the command line. Internal libraries within LLVM register parsers for
// flags, with no other way to configure them except pass these flags.
// To do this programmatically, we invoke ParseCommandLineOptions manually with
// a "fake argv".
// Note: setting flags with this method is stateful, since flags are just
// static globals within LLVM libraries.
void FeedLLVMWithFlags(const std::vector<std::string>& cl_opts) {
  std::vector<const char*> fake_argv = {""};
  for (const std::string& cl_opt : cl_opts) {
    fake_argv.push_back(cl_opt.c_str());
  }
  llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
}

using TargetModuleLinker = std::function<Status(
    llvm::Module*, const HloModuleConfig&, const std::string&)>;

Status LinkAndOptimizeModule(llvm::Module* module,
                             const HloModuleConfig& hlo_module_config,
                             llvm::Triple default_target_triple,
                             llvm::TargetMachine* target_machine,
                             int inline_threshold) {
  bool dump_ir = hlo_module_config.debug_options().xla_gpu_dump_llvmir();
  std::string outputs_dir;
  itex::io::GetTestUndeclaredOutputsDir(&outputs_dir);
  IrDumpingPassManager module_passes(module->getModuleIdentifier(), outputs_dir,
                                     dump_ir);

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  llvm::TargetLibraryInfoWrapperPass* tliwp =
      new llvm::TargetLibraryInfoWrapperPass(
          llvm::Triple(module->getTargetTriple()));
  module_passes.add(tliwp);

  // Try to fetch the target triple from the module. If not present, set a
  // default target triple.
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.getArch() == llvm::Triple::UnknownArch) {
    ITEX_VLOG(2) << "target triple not found in the module";
    target_triple = default_target_triple;
  }

  if (target_machine)
    module_passes.add(llvm::createTargetTransformInfoWrapperPass(
        target_machine->getTargetIRAnalysis()));

  // The LLVM IR verifier performs sanity checking on the IR. This helps
  // discover problems and report them in a meaningful manner, rather than let
  // later passes report obscure assertions because of unfulfilled invariants.
  module_passes.add(llvm::createVerifierPass());

  // Create the function-level pass manager. It needs data layout information
  // too.
  llvm::legacy::FunctionPassManager function_passes(module);

  int32_t opt_level =
      hlo_module_config.debug_options().xla_backend_optimization_level();

  if (opt_level < 2) {
    ITEX_LOG(ERROR) << std::string(80, '*');
    ITEX_LOG(ERROR) << "The XLA GPU backend doesn't support unoptimized code "
                       "generation but ";
    ITEX_LOG(ERROR) << "--xla_backend_optimization_level is set to "
                    << opt_level << "!";
    ITEX_LOG(ERROR) << "(Supported configuration is "
                       "--xla_backend_optimization_level >= 2.)";
    ITEX_LOG(ERROR) << std::string(80, '*');
  }

  // Add optimization passes, and set inliner threshold.
  AddOptimizationPasses(opt_level, 0, target_machine, &module_passes,
                        &function_passes, inline_threshold);

  // Loop unrolling exposes more opportunities for SROA. Therefore, we run SROA
  // again after the standard optimization passes [http://b/13329423].
  // TODO(jingyue): SROA may further expose more optimization opportunities such
  // as more precise alias analysis and more function inlining (SROA may change
  // the inlining cost of a function). For now, running SROA already emits good
  // enough code for the evaluated benchmarks. We may want to run more
  // optimizations later.
  if (opt_level > 0) {
    // LLVM's optimizer turns on SROA when the optimization level is greater
    // than 0. We mimic this behavior here.
    module_passes.add(llvm::createSROAPass());
  }

  // Verify that the module is well formed after optimizations ran.
  module_passes.add(llvm::createVerifierPass());

  // Done populating the pass managers. Now run them.

  function_passes.doInitialization();
  for (auto func = module->begin(); func != module->end(); ++func) {
    function_passes.run(*func);
  }
  function_passes.doFinalization();
  module_passes.run(*module);

  bool opt = true;
  itex::ReadBoolFromEnvVar("DPCPP_LLVM_OPT", true, &opt);
  if (opt) {
    RunOptimizationPipeline(module, target_machine);
  }

  std::string err;
  llvm::raw_string_ostream err_stream(err);

  // verifyModule() returns true if the module is broken.
  TF_RET_CHECK(!llvm::verifyModule(*module, &err_stream))
      << "Invalid LLVM IR after dpcpp optimizations:\n"
      << err_stream.str() << "\n";

  return Status::OK();
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void NVPTXBackendInit(const HloModuleConfig& hlo_module_config) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.

  // This flag tunes a threshold in branch folding. The default threshold, which
  // is one, is not suitable for CUDA programs where branches are more expensive
  // than for CPU programs. Setting the threshold to 2 improves the latency of
  // TwoDPatchDotProductKernel_IND_3_ND_48 by over 5%, and does not affect the
  // latency of other benchmarks so far.
  //
  // I also tried setting this threshold to other values:
  // * 3-6 gives similar results as 2;
  // * >6 start hurting the performance of at least dot product kernels.
  //
  // TODO(jingyue): The current threshold only considers the number of IR
  // instructions which do not accurately reflect the true cost. We need a
  // better cost model.
  FeedLLVMWithFlags({"-bonus-inst-threshold=2"});
  // Increase limit when scanning memory dependencies.  This helps to reduce
  // more redundant load instructions.
  //
  // The specific value is currently large enough for s3d in shoc benchmark,
  // which contains a lot of load instructions and many arithmetic instructions
  // between those loads.
  FeedLLVMWithFlags({"-memdep-block-scan-limit=500"});

  // intel llvm sycl opt flag.
  FeedLLVMWithFlags({"-sycl-opt=1"});

  llvm_ir::InitializeLLVMCommandLineOptions(
      hlo_module_config.debug_options().xla_backend_extra_options());

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

namespace {
StatusOr<std::string> EmitModuleToSpir(llvm::Module* module,
                                       const HloModuleConfig& module_config) {
  SPIRV::TranslatorOpts::ExtensionsStatusMap ExtensionsStatus;
  SPIRV::TranslatorOpts opts(SPIRV::VersionNumber::MaximumVersion,
                             ExtensionsStatus);
  opts.enableAllExtensions();  // enable all SPIR-V extension first

  std::ostringstream oss;
  std::string err;
  bool success = llvm::writeSpirv(module, opts, oss, err);
  if (!success) {
    return itex_xla::InternalError("Fails to convert LLVM as SPIR-V: %s", err);
  }
  return oss.str();
}
}  // namespace

namespace spir {
StatusOr<std::string> CompileToSpir(llvm::Module* module,
                                    const HloModuleConfig& hlo_module_config,
                                    const std::string& libdevice_dir_path) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, NVPTXBackendInit, hlo_module_config);

  std::string spir;
  {
    // itex::profiler::TraceMe activity(
    //     [&] { return absl::StrCat("Compiling IR:", module->getName().str());
    //     }, itex::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      ITEX_VLOG(2) << "Module '" << module->getName().str()
                   << "' is empty. Skipping compilation.";
      return std::string();
    }

    // No SPIR target machine?
    llvm::Triple default_target_triple("spir64-unknown-unknown");
    // std::unique_ptr<llvm::TargetMachine> target_machine =
    //     GetTargetMachine(default_target_triple, "generic",
    //                       hlo_module_config, "+ptx60");

    bool reuse = true;
    itex::ReadBoolFromEnvVar("TF_LLVM_OPT", true, &reuse);
    if (reuse) {
      // Link with libdevice, and optimize the LLVM module.
      TF_RETURN_IF_ERROR(LinkAndOptimizeModule(module, hlo_module_config,
                                               default_target_triple, nullptr,
                                               kDefaultInlineThreshold));
    }

    DumpStringToFileInDirOrStdout("module_opt.ll",
                                  llvm_ir::DumpModuleToString(*module),
                                  hlo_module_config.debug_options());

    // Lower optimized LLVM module to SPIR.
    TF_ASSIGN_OR_RETURN(spir, EmitModuleToSpir(module, hlo_module_config));
  }
  return spir;
}
}  // namespace spir

}  // namespace gpu
}  // namespace itex_xla
