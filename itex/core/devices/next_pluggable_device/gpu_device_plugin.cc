#include "itex/core/devices/next_pluggable_device/gpu_device_plugin.h"

#include "itex/core/utils/logging.h"
#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

int32_t TFNPD_GetDeviceCount(TF_Status* status) {
  int device_count;
  ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
  if (npdConfig.IfEnableNextPluggableDevice()) {
    ITEX_GPUGetDeviceCount(&device_count);
    return device_count;
  } else {
    return 0;
  }
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
  ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
  return ITEXSameDevicePjRtBufferCopy(src_buffer, c_client,
                                      npdConfig.isXlaAutoJitEnabled());
}

}  // namespace itex

#ifndef CC_BUILD
const TFNPD_Api* TFNPD_InitPlugin_Internal(TFNPD_PluginParams* params,
                                           TF_Status* tf_status) {
#else
const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params,
                                  TF_Status* tf_status) {
#endif
  ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
  params->struct_size = TFNPD_PLUGIN_PARAMS_STRUCT_SIZE;
  params->device_type =
      npdConfig.IfEnableNextPluggableDevice() ? "XPU" : "XPU_DUMMY";
  params->compilation_device_name = "XLA_GPU_JIT";
  params->is_pluggable_device = true;
  params->use_pjrt_on_demand_compile = false;
  params->priority = npdConfig.IfEnableNextPluggableDevice() ? 300 : 0;
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

#ifdef __cplusplus
extern "C" {
#endif
const PJRT_Api* GetPjrtApi();
const PJRT_Api* GetITEXPjrtApi();
#ifdef __cplusplus
}
#endif

#ifndef CC_BUILD
const PJRT_Api* GetPjrtApi_Internal() {
#else
const PJRT_Api* GetPjrtApi() {
#endif
  ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
  if (npdConfig.isXlaAutoJitEnabled()) {
    return GetPjrtApi();
  } else {
    return GetITEXPjrtApi();
  }
}
