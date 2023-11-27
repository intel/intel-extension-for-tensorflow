#include "itex/core/devices/next_pluggable_device/gpu_device_plugin.h"

#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

int32_t TFNPD_GetDeviceCount(TF_Status* status) {
  int device_count;
  ITEX_GPUGetDeviceCount(&device_count);
  return device_count;
}

void TFNPD_InitPluginInternalDeviceStates(TF_Status* status) {
  // TF_CreateAndSetPjRtCApiClient("XPU", status);
}

void TFNPD_XlaShapeToDeviceShapeRepresentation(
    XLA_Shape* serialized_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* serialized_device_shape,
    TF_Status* tf_status) {
  ITEXXlaShapeToDeviceShapeRepresentation(
      static_cast<void*>(serialized_xla_shape),
      static_cast<void*>(serialized_device_shape));
}

PJRT_Buffer* TFNPD_SameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                            PJRT_Client* c_client,
                                            TF_Status* status) {
  return ITEXSameDevicePjRtBufferCopy(src_buffer, c_client);
}

}  // namespace itex

#ifndef CC_BUILD
const TFNPD_Api* TFNPD_InitPlugin_Internal(TFNPD_PluginParams* params,
                                           TF_Status* tf_status) {
#else
const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params,
                                  TF_Status* tf_status) {
#endif
  params->struct_size = TFNPD_PLUGIN_PARAMS_STRUCT_SIZE;
  params->device_type = "XPU";
  params->compilation_device_name = "XLA_GPU_JIT";
  params->is_pluggable_device = true;
  params->use_pjrt_on_demand_compile = false;
  params->priority = 300;
  static TFNPD_Api tfnpd_api;

  tfnpd_api.TFNPD_GetDeviceCount = itex::TFNPD_GetDeviceCount;
  tfnpd_api.TFNPD_InitPluginInternalDeviceStates =
      itex::TFNPD_InitPluginInternalDeviceStates;
  tfnpd_api.TFNPD_XlaShapeToDeviceShapeRepresentation =
      itex::TFNPD_XlaShapeToDeviceShapeRepresentation;
  tfnpd_api.TFNPD_SameDevicePjRtBufferCopy =
      itex::TFNPD_SameDevicePjRtBufferCopy;
  return &tfnpd_api;
}
