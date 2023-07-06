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

/// Defines HS Dialect


#ifndef HS_DIALECT
#define HS_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"



//===----------------------------------------------------------------------===//
// HS dialect definitions
//===----------------------------------------------------------------------===//


def HSDialect : Dialect {
  let name = "hs";
  let summary = "This dialect encodes HS graph.";
  let description = [{
    This dialects models framework dependent graphs into a more abstract graph into HS-IR graph. 
    The HS-IR graph is used for searching for a decent sharding property.
  }];

  let extraClassDeclaration = [{

    // Preferred HSP attribute name key attached to the op
    static constexpr StringRef getHspAttrKey() { return "hs.hsp"; }

    static constexpr StringRef getDeviceInfoAttrKey() { return "hs.device_info"; }

    static std::string getHspAttrKeyFor(size_t result_id) {
      std::string key(getHspAttrKey().data());
      if (result_id > 0) {
        key.append("_" + std::to_string(result_id));
      }
      return key;
    }
  }];
  
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::hs";
}




#endif // HS_DIALECT