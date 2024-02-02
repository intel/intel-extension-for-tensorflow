#ifndef ITEX_CORE_DEVICES_NEXT_PLUGGABLE_DEVICE_GPU_DEVICE_PLUGIN_H_
#define ITEX_CORE_DEVICES_NEXT_PLUGGABLE_DEVICE_GPU_DEVICE_PLUGIN_H_

#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
const TFNPD_Api* TFNPD_InitPlugin_Internal(TFNPD_PluginParams* params,
                                           TF_Status* tf_status);
const PJRT_Api* GetPjrtApi_Internal();
#ifdef __cplusplus
}
#endif

#endif  // ITEX_CORE_DEVICES_NEXT_PLUGGABLE_DEVICE_GPU_DEVICE_PLUGIN_H_
