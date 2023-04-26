/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "gml_st/transforms/passes.h"
// #include "imex/Conversion/Passes.h"  // from @imex
// #include "imex/Transforms/Passes.h"  // from @imex
#include "itex/core/compiler/mlir/hlo/transforms/itex_gpu_passes.h"
#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {

#define GEN_PASS_DEF_GPUFUSIONREWRITEPASS
#include "transforms/itex_gpu_passes.h.inc"

namespace {
class GpuFusionRewritePass
    : public impl::GpuFusionRewritePassBase<GpuFusionRewritePass> {
 public:
  GpuFusionRewritePass() = default;
  using Pass::runPipeline;  // Give FusionRewritePattern access.

 private:
  void getDependentDialects(DialectRegistry& registry) const override;
  void runOnOperation() override;
};

// Rewrites `lmhlo.fusion` to `gpu.launch_func` for fusion regions that the
// HLO to GPU pipeline can handle.
class FusionRewritePattern : public OpRewritePattern<lmhlo::FusionOp> {
 public:
  explicit FusionRewritePattern(MLIRContext* ctx,
                                GpuFusionRewritePass& parentPass,
                                SymbolTable& symbolTable);

 private:
  LogicalResult matchAndRewrite(lmhlo::FusionOp fusionOp,
                                PatternRewriter& rewriter) const override;

  // Returns whether all ops in fusionOp's region are legal to rewritableTarget.
  bool isRewritable(lmhlo::FusionOp fusionOp) const;

  // Annotates gpu.launch_func with attribute specifying written operands.
  //
  // func.func @fusion(%arg0, %arg1 {lmhlo.written}) {
  //   gpu.launch_func args(%arg0, %arg1, %arg0)
  //
  // will add a `lmhlo.written = [false, true, false]` attribute.
  //
  // The 'written_operands' attribute is used later to retrieve which
  // gpu.launch_func arguments are written vs. just read.
  static void annotateLaunchFunc(func::FuncOp funcOp,
                                 PatternRewriter& rewriter);

  // Returns target where lowerable fusion ops are marked legal.
  static ConversionTarget getRewritableTarget(MLIRContext* ctx);

  GpuFusionRewritePass& parentPass;
  SymbolTable& symbolTable;
  ConversionTarget rewritableTarget = getRewritableTarget(getContext());
};
}  // namespace

namespace {
using namespace mlir;  // NOLINT
using ::mlir::func::FuncOp;
using ::mlir::gpu::GPUModuleOp;

// TODO(b/233761238): We only want to have this pipeline temporarily, as it is
// not yet clear how exactly it will look like. The goal is to merge this with
// the unified kernel generator + autofusion + XLA Next pipeline once we have
// it, and once this code stabilizes.
void createHloToGpuSpvPipeline(OpPassManager& pm,
                               ArrayRef<int64_t> blockTileDim,
                               ArrayRef<int64_t> warpTileDim,
                               ArrayRef<int64_t> threadTileDim) {
  pm.addNestedPass<FuncOp>(hlo::createUnbufferizePass());

  // HLO -> Linalg
  pm.addNestedPass<FuncOp>(mhlo::createLegalizeHloToLinalgPass());
  // TODO(b/244313563): This is a workaround to avoid temporary allocs within
  // threads. It works for as long as all of our operations are cwise. Vectorize
  // the inner loops instead.
  pm.addNestedPass<FuncOp>(createLinalgElementwiseOpFusionPass());

  // // Tiling
  // pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
  //     /*distribute=*/true, SmallVector<int64_t>(blockTileDim)));
  // pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
  //     /*distribute=*/true, SmallVector<int64_t>(warpTileDim)));
  // pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
  //     /*distribute=*/true, SmallVector<int64_t>(threadTileDim)));
  // pm.addNestedPass<FuncOp>(gml_st::createTilingReductionPass());
  // pm.addNestedPass<FuncOp>(createScalarizationPass());

  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(createCSEPass());
  // pm.addNestedPass<FuncOp>(gml_st::createComposeSetOpsPass());

  // Bufferization-related passes.
  pm.addNestedPass<FuncOp>(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(hlo::createOneShotBufferizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<FuncOp>(bufferization::createBufferDeallocationPass());
  pm.addNestedPass<FuncOp>(createGpuMapParallelLoopsPass());

  // Linalg + GmlSt -> GPU
  // pm.addNestedPass<FuncOp>(createGmlStToGpuPass());
  pm.addNestedPass<FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createParallelLoopToGpuPass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  constexpr llvm::StringRef kGpuDataLayoutSpec =
      "#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>";
  pm.addPass(createGpuKernelOutliningPass(kGpuDataLayoutSpec));
  pm.addNestedPass<GPUModuleOp>(createForLoopSpecializationPass());
  pm.addNestedPass<GPUModuleOp>(createLowerAffinePass());
  pm.addNestedPass<GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<GPUModuleOp>(createConvertSCFToCFPass());
  /*
  // pm.addPass(imex::createConvertGPUToGPUXPass());
  pm.addPass(imex::createSetSPIRVCapabilitiesPass());
  pm.addNestedPass<GPUModuleOp>(imex::createSetSPIRVAbiAttributePass());
  pm.addPass(imex::createConvertGPUXToSPIRVPass());
  OpPassManager& modulePm = pm.nest<spirv::ModuleOp>();
  modulePm.addPass(spirv::createLowerABIAttributesPass());
  modulePm.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

  // GPU -> low-level IR
  // Disable serialize here:
  // for mulit fusion, there may be duplicate kernel module names
  // they are renamed after this full pipeline. So we
  // serialize them separately.
  pm.addPass(imex::createSerializeSPIRVPass());
  pm.addNestedPass<FuncOp>(createReplaceAllocWithArgPass());
  */
}
}  // namespace

// Name of the 'gpu.launch_func' attribute which specifies the written operands.
static constexpr llvm::StringLiteral kWrittenOperandsAttrName = "lmhlo.written";

void GpuFusionRewritePass::getDependentDialects(
    DialectRegistry& registry) const {
  OpPassManager passManager;
  createHloToGpuSpvPipeline(passManager,
                            /*blockTileDim=*/{},
                            /*warpTileDim=*/{},
                            /*threadTileDim=*/{});
  passManager.getDependentDialects(registry);
}

void GpuFusionRewritePass::runOnOperation() {
  printf("--- Run on a ModuleOp!\n");
  SymbolTable symbolTable(getOperation());
  auto pattern =
      std::make_unique<FusionRewritePattern>(&getContext(), *this, symbolTable);
  mlir::FrozenRewritePatternSet patterns({&getContext(), std::move(pattern)});
  auto callback = [&](lmhlo::FusionOp fusion) {
    if (failed(applyOpPatternsAndFold(fusion, patterns)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  if (getOperation().walk(callback).wasInterrupted())
    return signalPassFailure();
  // Check if all lmhlo.fusion ops have been rewritten successfully
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    if (!func.getOps<lmhlo::FusionOp>().empty()) {
      printf("!!! Not all lmhlo.fusion legalized to gpu.launch_func\n");
      return signalPassFailure();
    }
  }
}

FusionRewritePattern::FusionRewritePattern(MLIRContext* ctx,
                                           GpuFusionRewritePass& parentPass,
                                           SymbolTable& symbolTable)
    : OpRewritePattern<lmhlo::FusionOp>::OpRewritePattern(ctx),
      parentPass(parentPass),
      symbolTable(symbolTable) {}

// Returns the number of elements each thread should handle for 'type'.
// The intention is that loads and stores are vectorized later on to this width
// to maximize memory throughput.
static int64_t getElementsPerThread(TensorType type) {
  // Don't vectorize if the number of elements cannot saturate the GPU.
  // Use a coarse heuristic because we don't know the target GPU here.
  const int64_t kNumFp32AlusOnV100 = 5376;
  if (type.getNumElements() < kNumFp32AlusOnV100) return 1;

  // Don't vectorize if element type is not int or float.
  if (!type.getElementType().isIntOrFloat()) return 1;

  // Vectorize so that loads and stores are 128 bits per thread.
  return 128 / type.getElementType().getIntOrFloatBitWidth();
}

// Returns the number of threads per block to use for 'type', given the number
// of elements each thread handles. The returned block size is in the [128, 384]
// range, preferrably close to 256 and evenly dividing the number of threads
// required to handle all elements in 'type'.
static int64_t getThreadsPerBlock(TensorType type, int64_t elementsPerThread) {
  int64_t numThreads =
      llvm::divideCeil(type.getNumElements(), elementsPerThread);

  // Use a single block for small problems.
  if (numThreads < 256) return numThreads;

  // Use 256 if that block size evenly divides the problem.
  if (numThreads % 256 == 0) return 256;

  int64_t elementSizeBits = 32;
  if (type.getElementType().isIntOrFloat())
    elementSizeBits = type.getElementType().getIntOrFloatBitWidth();
  int64_t threadSizeBits = elementSizeBits * elementsPerThread;

  // Search block sizes in the [128, 384] range near 256 with decreasing
  // power-of-2 factor, down to a multiple of a cache line (assumed to be 1024
  // bits). Use the first one that evenly divides the problem, which allows the
  // loop tail to be optimized away.
  for (int i = 128; i * threadSizeBits >= 1024; i /= 2) {
    // 2 * i: earlier iterations already handled even multiples of i.
    for (int blockSize = 256 - i; blockSize >= 128; blockSize -= 2 * i)
      if (numThreads % blockSize == 0) return blockSize;
    for (int blockSize = 256 + i; blockSize <= 384; blockSize += 2 * i)
      if (numThreads % blockSize == 0) return blockSize;
  }

  // None of the checked block sizes evenly divides the number of required
  // threads. Use a default of 256 and accept the loop tail.
  return 256;
}

LogicalResult FusionRewritePattern::matchAndRewrite(
    lmhlo::FusionOp fusionOp, PatternRewriter& rewriter) const {
  // If fusion_op (including its region) is not legal by rewriteable_target,
  // we expect lowering to GPU to fail or produce incorrect results.
  if (!isRewritable(fusionOp)) {
    return rewriter.notifyMatchFailure(fusionOp, "not rewritable");
  }

  // Collect values in fusion region defined above.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(fusionOp->getRegions(), captures);

  // // Converts statically shaped types to their 1D equivalent. This only works
  // // for element wise fusions and will have to become a more sophisticated
  // // pass when e.g. broadcasts are involved.
  // TypeConverter converter;
  // converter.addConversion([](Type type) { return type; });
  // converter.addConversion([](ShapedType type) {
  //   if (!type.hasStaticShape()) return type;
  //   return type.clone(type.getNumElements());
  // });
  // converter.addConversion([&](MemRefType type) {
  //   if (!type.hasStaticShape() || !type.getLayout().isIdentity()) return
  //   type; return MemRefType::get(type.getNumElements(),
  //   type.getElementType(),
  //                          MemRefLayoutAttrInterface(),
  //                          type.getMemorySpace());
  // });

  // Create a new module with a function, clone fusion region into it.
  Location loc = fusionOp.getLoc();
  auto moduleOp = rewriter.create<ModuleOp>(loc);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  // auto argTypes = llvm::to_vector(llvm::map_range(captures, [&](Value value)
  // {
  //   return converter.convertType(value.getType());
  // }));
  auto funcType =
      rewriter.getFunctionType(TypeRange(captures.getArrayRef()), llvm::None);
  auto funcOp = rewriter.create<func::FuncOp>(loc, "fusion", funcType);
  rewriter.setInsertionPointToEnd(funcOp.addEntryBlock());
  BlockAndValueMapping mapping;
  for (const auto& [from, to] :
       llvm::zip_first(captures, funcOp.getArguments())) {
    mapping.map(from, to);
  }
  rewriter.cloneRegionBefore(fusionOp.getRegion(), funcOp.getRegion(),
                             funcOp.end(), mapping);
  rewriter.mergeBlocks(&funcOp.back(), &funcOp.front());
  // // Convert statically shaped types to their 1D equivalent.
  // funcOp->walk([&](Operation* op) {
  //   for (auto result : op->getResults())
  //     result.setType(converter.convertType(result.getType()));
  // });
  // Add attribute to written function arguments.
  for (const BlockArgument& arg : funcOp.getArguments()) {
    if (llvm::any_of(arg.getUsers(), [](Operation* op) {
          return isa<memref::TensorStoreOp>(op);
        })) {
      funcOp.setArgAttr(arg.getArgNumber(), kWrittenOperandsAttrName,
                        rewriter.getUnitAttr());
    }
  }

  // Create and run the HLO to GPU pass pipeline.
  auto resultType =
      fusionOp.getFusionResults().front().getType().cast<TensorType>();
  int64_t elementsPerThread = getElementsPerThread(resultType);
  constexpr int64_t kThreadsPerWarp = 32;
  int64_t elementsPerWarp = elementsPerThread * kThreadsPerWarp;
  int64_t elementsPerBlock =
      getThreadsPerBlock(resultType, elementsPerThread) * elementsPerThread;
  // Note: passManager.enableIRPrinting() doesn't do anything on dynamic pass
  // pipelines. Printing needs to be enabled on the parent pass manager.
  PassManager passManager(getContext());
  applyPassManagerCLOptions(passManager);
  createHloToGpuSpvPipeline(passManager, {elementsPerBlock}, {elementsPerWarp},
                            {elementsPerThread});
  if (failed(parentPass.runPipeline(passManager, moduleOp)))
    return rewriter.notifyMatchFailure(fusionOp, "failed to run pipeline");

  // Clone the (single) gpu module with the device function.
  rewriter.setInsertionPoint(fusionOp->getParentOfType<func::FuncOp>());
  for (auto gpuModuleOp : moduleOp.getBodyRegion().getOps<gpu::GPUModuleOp>()) {
    StringAttr symbol =
        symbolTable.insert(rewriter.clone(*gpuModuleOp.getOperation()));
    if (failed(symbolTable.replaceAllSymbolUses(gpuModuleOp, symbol, funcOp)))
      return rewriter.notifyMatchFailure(fusionOp, "failed to replace symbol");
  }
  // Add 'gpu.container_module' attribute to parent module.
  fusionOp->getParentOfType<ModuleOp>()->setAttr(
      gpu::GPUDialect::getContainerModuleAttrName(), rewriter.getUnitAttr());

  // Annotate gpu.launch_func loc and attribute specifying written operands.
  funcOp->walk([&](gpu::LaunchFuncOp op) { op->setLoc(loc); });
  annotateLaunchFunc(funcOp, rewriter);

  // Replace fusion op with host function region.
  rewriter.splitBlock(&funcOp.front(),
                      funcOp.front().getTerminator()->getIterator());
  rewriter.mergeBlockBefore(&funcOp.front(), fusionOp, captures.getArrayRef());

  rewriter.eraseOp(fusionOp);
  rewriter.eraseOp(moduleOp);

  return success();
}

bool FusionRewritePattern::isRewritable(lmhlo::FusionOp fusionOp) const {
  if (fusionOp.getFusionResults().size() != 1)
    return false;  // Only rewrite fusion with a single result.
  if (isa<bufferization::ToTensorOp>(fusionOp.getFusionRoots().front()))
    return false;  // Don't rewrite empty (memcpy) fusion.
  auto callback = [this](Operation* op) {
    if (rewritableTarget.isLegal(op)) return WalkResult::advance();
    return WalkResult::interrupt();
  };
  return !fusionOp.getRegion().walk(callback).wasInterrupted();
}

void FusionRewritePattern::annotateLaunchFunc(func::FuncOp funcOp,
                                              PatternRewriter& rewriter) {
  funcOp.walk([&](gpu::LaunchFuncOp op) {
    auto writtenOperands = llvm::to_vector(
        llvm::map_range(op.getKernelOperands(), [&](Value operand) -> bool {
          auto arg = operand.dyn_cast<BlockArgument>();
          if (!arg) return false;
          return funcOp.getArgAttr(arg.getArgNumber(),
                                   kWrittenOperandsAttrName) != nullptr;
        }));
    op->setAttr(kWrittenOperandsAttrName,
                rewriter.getBoolArrayAttr(writtenOperands));
  });
}

// Returns whether 'type' is can be lowered by the FusionRewritePattern.
static bool isRewritableType(Type type) {
  auto shapedType = type.cast<ShapedType>();
  // Complex types are not yet supported.
  if (shapedType.getElementType().isa<ComplexType>()) {
    printf("!!! Complex Types are not supported\n");
    return false;
  }
  // Zero ranked shapes are not yet supported.
  if (shapedType.getRank() == 0) {
    printf("!!! 0 Rank tensor is not supported\n");
    return false;
  }
  // MemRef types need to have identity layout.
  if (auto memrefType = shapedType.dyn_cast<MemRefType>()) {
    if (!memrefType.getLayout().isIdentity()) {
      printf("!!! Memref need to have identity layout\n");
      return false;
    }
  }
  // Unsigned integers are not yet supported.
  if (auto intType = shapedType.getElementType().dyn_cast<IntegerType>()) {
    if (intType.isUnsigned()) {
      printf("!!! Unsigned integers are not supported\n");
      return false;
    }
  }
  return true;
}

ConversionTarget FusionRewritePattern::getRewritableTarget(MLIRContext* ctx) {
  ConversionTarget target(*ctx);
  // Mark expected auxiliary ops as legal.
  target.addLegalOp<lmhlo::TerminatorOp>();
  target.addDynamicallyLegalOp<bufferization::ToTensorOp>(
      [&](bufferization::ToTensorOp op) {
        return isRewritableType(op.getMemref().getType()) &&
               isRewritableType(op.getType());
      });
  target.addDynamicallyLegalOp<memref::TensorStoreOp>(
      [&](memref::TensorStoreOp op) {
        return isRewritableType(op.getTensor().getType()) &&
               isRewritableType(op.getMemref().getType());
      });
  // For now, use an explicit allow-list of hlo ops inside the fusion. If any
  // other op is present, the fusion will not be rewritten.
  target.addLegalOp<
      mhlo::AddOp, mhlo::AndOp, mhlo::AbsOp, mhlo::BroadcastOp,
      mhlo::BroadcastInDimOp, mhlo::CbrtOp, mhlo::CeilOp, mhlo::ConcatenateOp,
      mhlo::ConvertOp, mhlo::CompareOp, mhlo::CosineOp, mhlo::ConstantOp,
      mhlo::CopyOp, mhlo::DivOp, mhlo::DotOp, mhlo::DynamicUpdateSliceOp,
      mhlo::ExpOp, mhlo::Expm1Op, mhlo::FloorOp, mhlo::IotaOp, mhlo::LogOp,
      mhlo::Log1pOp, mhlo::LogisticOp, mhlo::MulOp, mhlo::NegOp, mhlo::PadOp,
      mhlo::PowOp, mhlo::ReduceOp, mhlo::ReshapeOp, mhlo::ReturnOp,
      mhlo::RoundOp, mhlo::RsqrtOp, mhlo::SelectOp, mhlo::SignOp, mhlo::SineOp,
      mhlo::SliceOp, mhlo::SqrtOp, mhlo::SubtractOp, mhlo::TanhOp,
      mhlo::TransposeOp>();
  return target;
}

std::unique_ptr<OperationPass<ModuleOp>> createGpuFusionRewritePass() {
  return std::make_unique<GpuFusionRewritePass>();
}

ArrayAttr getWrittenOperandsAttribute(Operation* op) {
  return op->getAttrOfType<ArrayAttr>(kWrittenOperandsAttrName);
}

}  // namespace mlir
