/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "itex/core/compiler/mlir/xla/hlo_utils.h"

#include <string>
#include <vector>

#include "itex/core/compiler/xla/literal.h"
#include "itex/core/utils/logging.h"
#include "lhlo/IR/lhlo_ops.h"
#include "mlir/IR/AffineMap.h"      // from @llvm-project
#include "mlir/IR/Attributes.h"     // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"   // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project

namespace itex_xla {
namespace {

using itex_xla::LiteralBase;
using itex_xla::StatusOr;
using mlir::AffineMap;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::ShapedType;

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFromLiteral(
    const ShapedType& type, const LiteralBase& literal) {
  auto data_span = literal.data<CppType>();
  return ::mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(data_span.data(), data_span.size()));
}

StatusOr<AffineMap> GetPermutationIfAvailable(const Shape& shape,
                                              mlir::Builder builder) {
  // N.B. IsMonotonicWithDim0Major ignores tiling, and I can't change it because
  // some XLA code relies on it treating tiled layouts as equivalent to untiled
  // layouts, so the check to rule out tiling has to come /before/ the
  // early-return branch, or we'd miss tiled monotonic layouts.
  if (!shape.layout().tiles().empty()) {
    return itex::errors::Internal("Tiled layouts are not yet supported");
  }
  if (!shape.has_layout() ||
      LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return AffineMap();
  }
  if (!shape.is_static()) {
    return itex::errors::Internal(
        "Permutations for dynamic shapes are not yet supported");
  }
  int64_t accumulated_stride = 1;
  llvm::SmallVector<int64_t, 4> strides(shape.rank(), 1);
  for (int64_t dim : LayoutUtil::MinorToMajor(shape)) {
    strides[dim] = accumulated_stride;
    accumulated_stride *= shape.dimensions(dim);
  }
  if (accumulated_stride == 0) {
    return AffineMap();
  }
  return makeStridedLinearLayoutMap(strides, /*offset=*/0,
                                    builder.getContext());
}

template <typename T>
void CopyDenseElementsBy(mlir::DenseElementsAttr data,
                         std::vector<uint8_t>* output) {
  output->resize(data.getNumElements() * sizeof(T));
  int i = 0;
  for (T element : data.getValues<T>()) {
    std::memcpy(&(*output)[i], &element, sizeof(T));
    i += sizeof(T);
  }
}

}  // namespace

StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder) {
  auto element_type_or =
      ConvertPrimitiveTypeToMLIRType(shape.element_type(), builder);
  if (!element_type_or.ok()) return element_type_or.status();

  using mlir::MemRefType;
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());
  auto permutation_or = GetPermutationIfAvailable(shape, builder);
  if (!permutation_or.ok()) return permutation_or.status();
  return MemRefType::get(array, element_type_or.ValueOrDie(),
                         permutation_or.ValueOrDie());
}

StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, Builder builder) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorShapeToType<mlir::RankedTensorType>(
                          literal.shape(), builder));

  // TODO(hinsu): Support remaining XLA primitive types.
  auto element_type = literal.shape().element_type();
  switch (element_type) {
    case PrimitiveType::PRED:
      return CreateDenseAttrFromLiteral<bool>(type, literal);
    case PrimitiveType::F16:
      return CreateDenseAttrFromLiteral<half>(type, literal);
    case PrimitiveType::BF16:
      return CreateDenseAttrFromLiteral<bfloat16>(type, literal);
    case PrimitiveType::F32:
      return CreateDenseAttrFromLiteral<float>(type, literal);
    case PrimitiveType::F64:
      return CreateDenseAttrFromLiteral<double>(type, literal);
    case PrimitiveType::S8:
      return CreateDenseAttrFromLiteral<int8_t>(type, literal);
    case PrimitiveType::S16:
      return CreateDenseAttrFromLiteral<int16_t>(type, literal);
    case PrimitiveType::S32:
      return CreateDenseAttrFromLiteral<int32_t>(type, literal);
    case PrimitiveType::S64:
      return CreateDenseAttrFromLiteral<int64_t>(type, literal);
    case PrimitiveType::U8:
      return CreateDenseAttrFromLiteral<uint8_t>(type, literal);
    case PrimitiveType::U16:
      return CreateDenseAttrFromLiteral<uint16_t>(type, literal);
    case PrimitiveType::U32:
      return CreateDenseAttrFromLiteral<uint32_t>(type, literal);
    case PrimitiveType::U64:
      return CreateDenseAttrFromLiteral<uint64_t>(type, literal);
    case PrimitiveType::C64:
      return CreateDenseAttrFromLiteral<complex64>(type, literal);
    case PrimitiveType::C128:
      return CreateDenseAttrFromLiteral<complex128>(type, literal);
    default:
      return itex::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

Status CopyDenseElementsDataToXlaFormat(mlir::DenseElementsAttr data,
                                        std::vector<uint8_t>* output) {
  mlir::Type element_type = data.getType().getElementType();

  // TODO(hinsu): Support remaining XLA primitive types.
  if (element_type.isInteger(1)) {
    CopyDenseElementsBy<bool>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(8)) {
    CopyDenseElementsBy<uint8_t>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(16)) {
    CopyDenseElementsBy<uint16_t>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(32)) {
    CopyDenseElementsBy<uint32_t>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(64)) {
    CopyDenseElementsBy<uint64_t>(data, output);
    return Status::OK();
  }
  if (element_type.isBF16()) {
    CopyDenseElementsBy<bfloat16>(data, output);
    return Status::OK();
  }
  if (element_type.isF16()) {
    CopyDenseElementsBy<half>(data, output);
    return Status::OK();
  }
  if (element_type.isF32()) {
    CopyDenseElementsBy<float>(data, output);
    return Status::OK();
  }
  if (element_type.isF64()) {
    CopyDenseElementsBy<double>(data, output);
    return Status::OK();
  }
  if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>()) {
    if (complex_type.getElementType().isF32()) {
      CopyDenseElementsBy<complex64>(data, output);
      return Status::OK();
    }
    if (complex_type.getElementType().isF64()) {
      CopyDenseElementsBy<complex128>(data, output);
      return Status::OK();
    }
  }
  return itex::errors::Internal(
      "Unsupported type in CopyDenseElementsDataToXlaFormat");
}

StatusOr<int> GetElementTypeBytes(mlir::Type type) {
  if (type.isInteger(1)) {
    return 1;
  }
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    TF_ASSIGN_OR_RETURN(int bytes,
                        GetElementTypeBytes(complex_type.getElementType()));
    return bytes * 2;
  }
  int width = type.getIntOrFloatBitWidth();
  TF_RET_CHECK(width % 8 == 0);
  return width / 8;
}

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64_t> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(shape.empty() ? vector.size() : shape,
                                  builder.getIntegerType(64)),
      vector);
}

StatusOr<mlir::Type> ConvertPrimitiveTypeToMLIRType(PrimitiveType element_type,
                                                    mlir::Builder builder) {
  switch (element_type) {
    case PrimitiveType::PRED:
      return builder.getI1Type();
    case PrimitiveType::F16:
      return builder.getF16Type();
    case PrimitiveType::BF16:
      return builder.getBF16Type();
    case PrimitiveType::F32:
      return builder.getF32Type();
    case PrimitiveType::F64:
      return builder.getF64Type();
    case PrimitiveType::S8:
      return builder.getIntegerType(8);
    case PrimitiveType::S16:
      return builder.getIntegerType(16);
    case PrimitiveType::S32:
      return builder.getIntegerType(32);
    case PrimitiveType::S64:
      return builder.getIntegerType(64);
    case PrimitiveType::U8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case PrimitiveType::U16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case PrimitiveType::U32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case PrimitiveType::U64:
      return builder.getIntegerType(64, /*isSigned=*/false);
    case PrimitiveType::C64:
      return mlir::ComplexType::get(builder.getF32Type());
    case PrimitiveType::C128:
      return mlir::ComplexType::get(builder.getF64Type());
    // TODO(b/130356985): Support unsigned primitive types.
    default:
      return itex::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

mlir::mhlo::GatherDimensionNumbersAttr CreateGatherDimensionNumbers(
    const GatherDimensionNumbers& input, mlir::Builder builder) {
  auto get_i64_array = [](absl::Span<const int64_t> container) {
    return llvm::ArrayRef<int64_t>{container.data(), container.size()};
  };
  return mlir::mhlo::GatherDimensionNumbersAttr::get(
      builder.getContext(), get_i64_array(input.offset_dims()),
      get_i64_array(input.collapsed_slice_dims()),
      get_i64_array(input.start_index_map()), input.index_vector_dim());
}

StatusOr<::itex_xla::HloOpcode> MhloToHloOpcode(mlir::Operation* op) {
  using mlir::isa;

  if (isa<mlir::mhlo::ConstantOp, mlir::lmhlo::ConstantOp>(op)) {
    return itex_xla::HloOpcode::kConstant;
  } else if (isa<mlir::mhlo::IotaOp, mlir::lmhlo::IotaOp>(op)) {
    return itex_xla::HloOpcode::kIota;
  } else if (isa<mlir::mhlo::ConvertOp, mlir::lmhlo::ConvertOp>(op)) {
    return itex_xla::HloOpcode::kConvert;
  } else if (isa<mlir::mhlo::AddOp, mlir::lmhlo::AddOp>(op)) {
    return itex_xla::HloOpcode::kAdd;
  } else if (isa<mlir::mhlo::Atan2Op, mlir::lmhlo::Atan2Op>(op)) {
    return itex_xla::HloOpcode::kAtan2;
  } else if (isa<mlir::mhlo::DivOp, mlir::lmhlo::DivOp>(op)) {
    return itex_xla::HloOpcode::kDivide;
  } else if (isa<mlir::mhlo::MaxOp, mlir::lmhlo::MaxOp>(op)) {
    return itex_xla::HloOpcode::kMaximum;
  } else if (isa<mlir::mhlo::MinOp, mlir::lmhlo::MinOp>(op)) {
    return itex_xla::HloOpcode::kMinimum;
  } else if (isa<mlir::mhlo::MulOp, mlir::lmhlo::MulOp>(op)) {
    return itex_xla::HloOpcode::kMultiply;
  } else if (isa<mlir::mhlo::PowOp, mlir::lmhlo::PowOp>(op)) {
    return itex_xla::HloOpcode::kPower;
  } else if (isa<mlir::mhlo::RemOp, mlir::lmhlo::RemOp>(op)) {
    return itex_xla::HloOpcode::kRemainder;
  } else if (isa<mlir::mhlo::ShiftLeftOp, mlir::lmhlo::ShiftLeftOp>(op)) {
    return itex_xla::HloOpcode::kShiftLeft;
  } else if (isa<mlir::mhlo::ShiftRightArithmeticOp,
                 mlir::lmhlo::ShiftRightArithmeticOp>(op)) {
    return itex_xla::HloOpcode::kShiftRightArithmetic;
  } else if (isa<mlir::mhlo::ShiftRightLogicalOp,
                 mlir::lmhlo::ShiftRightLogicalOp>(op)) {
    return itex_xla::HloOpcode::kShiftRightLogical;
  } else if (isa<mlir::mhlo::SubtractOp, mlir::lmhlo::SubtractOp>(op)) {
    return itex_xla::HloOpcode::kSubtract;
  } else if (isa<mlir::mhlo::XorOp, mlir::lmhlo::XorOp>(op)) {
    return itex_xla::HloOpcode::kXor;
  } else if (isa<mlir::mhlo::InfeedOp, mlir::lmhlo::InfeedOp>(op)) {
    return itex_xla::HloOpcode::kInfeed;
  } else if (isa<mlir::mhlo::OutfeedOp, mlir::lmhlo::OutfeedOp>(op)) {
    return itex_xla::HloOpcode::kOutfeed;
  } else if (isa<mlir::mhlo::SendOp>(op)) {
    return itex_xla::HloOpcode::kSend;
  } else if (isa<mlir::mhlo::RecvOp>(op)) {
    return itex_xla::HloOpcode::kRecv;
  } else if (isa<mlir::mhlo::ReplicaIdOp, mlir::lmhlo::ReplicaIdOp>(op)) {
    return itex_xla::HloOpcode::kReplicaId;
  } else if (isa<mlir::mhlo::AfterAllOp>(op)) {
    return itex_xla::HloOpcode::kAfterAll;
  } else if (isa<mlir::mhlo::AllReduceOp, mlir::lmhlo::AllReduceOp>(op)) {
    return itex_xla::HloOpcode::kAllReduce;
  } else if (isa<mlir::mhlo::AllToAllOp>(op)) {
    return itex_xla::HloOpcode::kAllToAll;
  } else if (isa<mlir::mhlo::TupleOp>(op)) {
    return itex_xla::HloOpcode::kTuple;
  } else if (isa<mlir::mhlo::BatchNormGradOp, mlir::lmhlo::BatchNormGradOp>(
                 op)) {
    return itex_xla::HloOpcode::kBatchNormGrad;
  } else if (isa<mlir::mhlo::BatchNormInferenceOp,
                 mlir::lmhlo::BatchNormInferenceOp>(op)) {
    return itex_xla::HloOpcode::kBatchNormInference;
  } else if (isa<mlir::mhlo::BatchNormTrainingOp,
                 mlir::lmhlo::BatchNormTrainingOp>(op)) {
    return itex_xla::HloOpcode::kBatchNormTraining;
  } else if (isa<mlir::mhlo::BitcastConvertOp, mlir::lmhlo::BitcastConvertOp>(
                 op)) {
    return itex_xla::HloOpcode::kBitcastConvert;
  } else if (isa<mlir::mhlo::BroadcastOp, mlir::lmhlo::BroadcastOp>(op)) {
    return itex_xla::HloOpcode::kBroadcast;
  } else if (isa<mlir::mhlo::CholeskyOp, mlir::lmhlo::CholeskyOp>(op)) {
    return itex_xla::HloOpcode::kCholesky;
  } else if (isa<mlir::mhlo::ClampOp, mlir::lmhlo::ClampOp>(op)) {
    return itex_xla::HloOpcode::kClamp;
  } else if (isa<mlir::mhlo::ConcatenateOp, mlir::lmhlo::ConcatenateOp>(op)) {
    return itex_xla::HloOpcode::kConcatenate;
  } else if (isa<mlir::mhlo::ConvolutionOp, mlir::lmhlo::ConvolutionOp>(op)) {
    return itex_xla::HloOpcode::kConvolution;
  } else if (isa<mlir::mhlo::SortOp, mlir::lmhlo::SortOp>(op)) {
    return itex_xla::HloOpcode::kSort;
  } else if (isa<mlir::mhlo::RngBitGeneratorOp>(op)) {
    return itex_xla::HloOpcode::kRngBitGenerator;
  } else if (isa<mlir::mhlo::XlaRngGetAndUpdateStateOp>(op)) {
    return itex_xla::HloOpcode::kRngGetAndUpdateState;
  } else if (isa<mlir::mhlo::FusionOp, mlir::lmhlo::FusionOp>(op)) {
    return itex_xla::HloOpcode::kFusion;
  } else if (isa<mlir::mhlo::BitcastOp>(op)) {
    return itex_xla::HloOpcode::kBitcast;
  } else if (isa<mlir::mhlo::AbsOp, mlir::lmhlo::AbsOp>(op)) {
    return itex_xla::HloOpcode::kAbs;
  } else if (isa<mlir::mhlo::CbrtOp, mlir::lmhlo::CbrtOp>(op)) {
    return itex_xla::HloOpcode::kCbrt;
  } else if (isa<mlir::mhlo::CeilOp, mlir::lmhlo::CeilOp>(op)) {
    return itex_xla::HloOpcode::kCeil;
  } else if (isa<mlir::mhlo::ClzOp, mlir::lmhlo::ClzOp>(op)) {
    return itex_xla::HloOpcode::kClz;
  } else if (isa<mlir::mhlo::CosineOp, mlir::lmhlo::CosineOp>(op)) {
    return itex_xla::HloOpcode::kCos;
  } else if (isa<mlir::mhlo::ExpOp, mlir::lmhlo::ExpOp>(op)) {
    return itex_xla::HloOpcode::kExp;
  } else if (isa<mlir::mhlo::Expm1Op, mlir::lmhlo::Expm1Op>(op)) {
    return itex_xla::HloOpcode::kExpm1;
  } else if (isa<mlir::mhlo::FloorOp, mlir::lmhlo::FloorOp>(op)) {
    return itex_xla::HloOpcode::kFloor;
  } else if (isa<mlir::mhlo::ImagOp, mlir::lmhlo::ImagOp>(op)) {
    return itex_xla::HloOpcode::kImag;
  } else if (isa<mlir::mhlo::IsFiniteOp, mlir::lmhlo::IsFiniteOp>(op)) {
    return itex_xla::HloOpcode::kIsFinite;
  } else if (isa<mlir::mhlo::LogOp, mlir::lmhlo::LogOp>(op)) {
    return itex_xla::HloOpcode::kLog;
  } else if (isa<mlir::mhlo::Log1pOp, mlir::lmhlo::Log1pOp>(op)) {
    return itex_xla::HloOpcode::kLog1p;
  } else if (isa<mlir::mhlo::LogisticOp>(op)) {
    return itex_xla::HloOpcode::kLogistic;
  } else if (isa<mlir::mhlo::NotOp, mlir::lmhlo::NotOp>(op)) {
    return itex_xla::HloOpcode::kNot;
  } else if (isa<mlir::mhlo::NegOp, mlir::lmhlo::NegOp>(op)) {
    return itex_xla::HloOpcode::kNegate;
  } else if (isa<mlir::mhlo::PopulationCountOp, mlir::lmhlo::PopulationCountOp>(
                 op)) {
    return itex_xla::HloOpcode::kPopulationCount;
  } else if (isa<mlir::mhlo::RealOp, mlir::lmhlo::RealOp>(op)) {
    return itex_xla::HloOpcode::kReal;
  } else if (isa<mlir::mhlo::RoundOp, mlir::lmhlo::RoundOp>(op)) {
    return itex_xla::HloOpcode::kRoundNearestAfz;
  } else if (isa<mlir::mhlo::RoundNearestEvenOp,
                 mlir::lmhlo::RoundNearestEvenOp>(op)) {
    return itex_xla::HloOpcode::kRoundNearestEven;
  } else if (isa<mlir::mhlo::RsqrtOp, mlir::lmhlo::RsqrtOp>(op)) {
    return itex_xla::HloOpcode::kRsqrt;
  } else if (isa<mlir::mhlo::SignOp, mlir::lmhlo::SignOp>(op)) {
    return itex_xla::HloOpcode::kSign;
  } else if (isa<mlir::mhlo::SineOp, mlir::lmhlo::SineOp>(op)) {
    return itex_xla::HloOpcode::kSin;
  } else if (isa<mlir::mhlo::SqrtOp, mlir::lmhlo::SqrtOp>(op)) {
    return itex_xla::HloOpcode::kSqrt;
  } else if (isa<mlir::mhlo::TanOp, mlir::lmhlo::TanOp>(op)) {
    return itex_xla::HloOpcode::kTan;
  } else if (isa<mlir::mhlo::TanhOp, mlir::lmhlo::TanhOp>(op)) {
    return itex_xla::HloOpcode::kTanh;
  } else if (isa<mlir::mhlo::ComplexOp, mlir::lmhlo::ComplexOp>(op)) {
    return itex_xla::HloOpcode::kComplex;
  } else if (isa<mlir::mhlo::AndOp, mlir::lmhlo::AndOp>(op)) {
    return itex_xla::HloOpcode::kAnd;
  } else if (isa<mlir::mhlo::OrOp, mlir::lmhlo::OrOp>(op)) {
    return itex_xla::HloOpcode::kOr;
  } else if (isa<mlir::mhlo::WhileOp, mlir::lmhlo::WhileOp>(op)) {
    return itex_xla::HloOpcode::kWhile;
  } else if (isa<mlir::mhlo::ReduceOp, mlir::lmhlo::ReduceOp>(op)) {
    return itex_xla::HloOpcode::kReduce;
  } else if (isa<mlir::mhlo::GetTupleElementOp>(op)) {
    return itex_xla::HloOpcode::kGetTupleElement;
  } else if (isa<mlir::mhlo::CompareOp, mlir::lmhlo::CompareOp>(op)) {
    return itex_xla::HloOpcode::kCompare;
  } else if (isa<mlir::mhlo::SliceOp, mlir::lmhlo::SliceOp>(op)) {
    return itex_xla::HloOpcode::kSlice;
  } else if (isa<mlir::mhlo::DynamicSliceOp, mlir::lmhlo::DynamicSliceOp>(op)) {
    return itex_xla::HloOpcode::kDynamicSlice;
  } else if (isa<mlir::mhlo::DynamicUpdateSliceOp,
                 mlir::lmhlo::DynamicUpdateSliceOp>(op)) {
    return itex_xla::HloOpcode::kDynamicUpdateSlice;
  } else if (isa<mlir::mhlo::CollectivePermuteOp,
                 mlir::lmhlo::CollectivePermuteOp>(op)) {
    return itex_xla::HloOpcode::kCollectivePermute;
  } else if (isa<mlir::mhlo::CopyOp, mlir::lmhlo::CopyOp>(op)) {
    return itex_xla::HloOpcode::kCopy;
  } else if (isa<mlir::mhlo::CustomCallOp, mlir::lmhlo::CustomCallOp>(op)) {
    return itex_xla::HloOpcode::kCustomCall;
  } else if (isa<mlir::mhlo::DotOp, mlir::lmhlo::DotOp>(op)) {
    return itex_xla::HloOpcode::kDot;
  } else if (isa<mlir::mhlo::FftOp, mlir::lmhlo::FftOp>(op)) {
    return itex_xla::HloOpcode::kFft;
  } else if (isa<mlir::mhlo::GatherOp, mlir::lmhlo::GatherOp>(op)) {
    return itex_xla::HloOpcode::kGather;
  } else if (isa<mlir::mhlo::GetDimensionSizeOp>(op)) {
    return itex_xla::HloOpcode::kGetDimensionSize;
  } else if (isa<mlir::mhlo::MapOp, mlir::lmhlo::MapOp>(op)) {
    return itex_xla::HloOpcode::kMap;
  } else if (isa<mlir::mhlo::ReshapeOp, mlir::lmhlo::ReshapeOp>(op)) {
    return itex_xla::HloOpcode::kReshape;
  } else if (isa<mlir::mhlo::DynamicReshapeOp>(op)) {
    return itex_xla::HloOpcode::kDynamicReshape;
  } else if (isa<mlir::mhlo::ScatterOp, mlir::lmhlo::ScatterOp>(op)) {
    return itex_xla::HloOpcode::kScatter;
  } else if (isa<mlir::mhlo::SelectOp, mlir::lmhlo::SelectOp>(op)) {
    return itex_xla::HloOpcode::kSelect;
  } else if (isa<mlir::mhlo::SelectAndScatterOp,
                 mlir::lmhlo::SelectAndScatterOp>(op)) {
    return itex_xla::HloOpcode::kSelectAndScatter;
  } else if (isa<mlir::mhlo::SetDimensionSizeOp>(op)) {
    return itex_xla::HloOpcode::kSetDimensionSize;
  } else if (isa<mlir::mhlo::ReverseOp, mlir::lmhlo::ReverseOp>(op)) {
    return itex_xla::HloOpcode::kReverse;
  } else if (isa<mlir::mhlo::PadOp, mlir::lmhlo::PadOp>(op)) {
    return itex_xla::HloOpcode::kPad;
  } else if (isa<mlir::mhlo::TransposeOp, mlir::lmhlo::TransposeOp>(op)) {
    return itex_xla::HloOpcode::kTranspose;
  } else if (isa<mlir::mhlo::TriangularSolveOp, mlir::lmhlo::TriangularSolveOp>(
                 op)) {
    return itex_xla::HloOpcode::kTriangularSolve;
  } else if (isa<mlir::mhlo::ReduceWindowOp, mlir::lmhlo::ReduceWindowOp>(op)) {
    return itex_xla::HloOpcode::kReduceWindow;
  } else if (isa<mlir::mhlo::ReducePrecisionOp, mlir::lmhlo::ReducePrecisionOp>(
                 op)) {
    return itex_xla::HloOpcode::kReducePrecision;
  } else if (isa<mlir::mhlo::DotGeneralOp>(op)) {
    return itex_xla::HloOpcode::kDot;
  } else if (isa<mlir::mhlo::BroadcastInDimOp, mlir::lmhlo::BroadcastInDimOp>(
                 op)) {
    return itex_xla::HloOpcode::kBroadcast;
  } else {
    std::string s;
    {
      llvm::raw_string_ostream os(s);
      op->print(os);
    }
    return itex::errors::Unimplemented("Unimplemented MHLO -> HloOpcode: ", s);
  }
}

}  // namespace itex_xla
