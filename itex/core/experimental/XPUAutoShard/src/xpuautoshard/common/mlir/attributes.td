
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

#ifndef HS_ATTRIBUTES
#define HS_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "xpuautoshard/common/mlir/dialect.td"

// Base class for HS dialect attributes.
class HS_Attr<string name> : AttrDef<HSDialect, name>;

def ShardingPropertyAttr : HS_Attr<"ShardingProperty"> {
  let mnemonic = "hsp";
  let parameters = (ins
    "as::ShardingPropertyRef": $prop
  );
  let extraClassDeclaration = [{
    as::ShardingPropertyRef getShardingProperty() const;
    bool isInitialized() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def DeviceInfoAttr : HS_Attr<"DeviceInfo"> {
  let mnemonic = "device_info";
  let description = [{
    Device info attr attached to the root op of a framework DAG.
  }];

  let parameters = (ins
    "as::DeviceInfo": $info
  );

  let extraClassDeclaration = [{
    const as::DeviceInfo& getDeviceInfo() const;
  }];
  let hasCustomAssemblyFormat = 1;
}


#endif // HS_ATTRIBUTES