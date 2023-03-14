/* Copyright (c) 2023 Intel Corporation

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
#include "xpuautoshard/common/mlir/attributes.h"

#include <vector>

#include "llvm/ADT/ArrayRef.h"               // from @llvm-project
#include "llvm/ADT/STLExtras.h"              // from @llvm-project
#include "llvm/ADT/TypeSwitch.h"             // from @llvm-project
#include "llvm/Support/Debug.h"              // from @llvm-project
#include "llvm/Support/ErrorHandling.h"      // from @llvm-project
#include "llvm/Support/SMLoc.h"              // from @llvm-project
#include "llvm/Support/raw_ostream.h"        // from @llvm-project
#include "mlir/IR/AsmState.h"                // from @llvm-project
#include "mlir/IR/Attributes.h"              // from @llvm-project
#include "mlir/IR/Builders.h"                // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/IR/Dialect.h"                 // from @llvm-project
#include "mlir/IR/DialectImplementation.h"   // from @llvm-project
#include "mlir/IR/FunctionImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/IR/OperationSupport.h"        // from @llvm-project
#include "mlir/IR/TypeRange.h"               // from @llvm-project
#include "mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/Support/LogicalResult.h"      // from @llvm-project
#include "xpuautoshard/common/mlir/dialect.h"

namespace mlir {
namespace hs {
using as::ShardingPropertyRef;
using as::SplitSpec;

::mlir::Attribute ShardingPropertyAttr::parse(::mlir::AsmParser& odsParser,
                                              ::mlir::Type odsType) {
  // TODO(itex) : Finish this

  return {};
}
void ShardingPropertyAttr::print(::mlir::AsmPrinter& printer) const {
  auto print_split_specs = [](llvm::raw_ostream& os,
                              const std::vector<SplitSpec>& split_specs) {
    for (size_t i = 0; i < split_specs.size(); i++) {
      if (split_specs[i].isInitialized()) {
        os << " [" << i << "] by ";
        switch (split_specs[i].getType()) {
          case SplitSpec::SplitType::RATIO:
            os << "ratios(";
            for (auto ratio : split_specs[i].getRatios()) {
              os << ratio << ",";
            }
            os << ")";
            break;
          case SplitSpec::SplitType::SIZE:
            os << "sizes(";
            for (auto size : split_specs[i].getSizes()) {
              os << size << ",";
            }
            os << ")";
            if (split_specs[i].isMultiStages()) {
              auto stage_nums = split_specs[i].getStageNums();
              os << ":stage_nums(";
              for (size_t i = 0; i < stage_nums.size(); i++) {
                os << stage_nums[i] << ",";
              }
              os << ")";
            }
            break;
          case SplitSpec::SplitType::SINGLE:
            os << "single";
            break;
          default:
            os << "unknown";
            break;
        }
      }
    }
  };
  auto print_post_ops = [](llvm::raw_ostream& os, ShardingPropertyRef prop) {
    if (!prop || prop->getNumPostOps() == 0) {
      return;
    }
    os << ", PostOps:[";
    for (auto&& post_op : prop->getPostOps()) {
      os << post_op->getName() << ",";
    }
    os << "]";
  };
  llvm::raw_ostream& os = printer.getStream();
  auto&& prop = getShardingProperty();
  if (!prop) {
    os << "uninited:<null>";
  } else if (!prop->isInitialized()) {
    os << "uninited:";
    print_split_specs(os, prop->getSplitSpecs());
  } else {
    if (prop->isSplitSingleOnly()) {
      os << "single_split_only:[";
      auto offsets = prop->getStageOffsets();
      for (auto offset : offsets) {
        os << offset << ",";
      }
      os << "]";
      if (prop->isShapeTensor()) {
        os << ",shape_splits[";
        print_split_specs(os, prop->getShapeSplitSpecs());
        os << "]";
      } else if (prop->isSizeTensor()) {
        os << ",size_split[";
        print_split_specs(os, {prop->getSizeSplitSpec()});
        os << "]";
      }
    } else {
      os << "split:";
      print_split_specs(os, prop->getSplitSpecs());
    }
    print_post_ops(os, prop);
  }
}

::mlir::Attribute DeviceInfoAttr::parse(::mlir::AsmParser& odsParser,
                                        ::mlir::Type odsType) {
  // TODO(itex) : Finish this

  return {};
}
void DeviceInfoAttr::print(::mlir::AsmPrinter& printer) const {
  auto&& info = getDeviceInfo();
  llvm::raw_ostream& os = printer.getStream();
  for (auto device : info.getDevices()) {
    os << "[" << device.getId() << "]" << device.getName() << ":"
       << device.getScore() << ",";
  }
  // TODO(itex) : finish this

  //  llvm::raw_ostream &os = printer.getStream();
  //   os << "DeviceInfoAttr print";
}

}  // namespace hs

}  // namespace mlir
