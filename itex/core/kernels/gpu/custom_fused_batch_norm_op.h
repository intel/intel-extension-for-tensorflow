#ifndef ITEX_CORE_KERNELS_GPU_CUSTOM_FUSED_BATCH_NORM_OP_H_
#define ITEX_CORE_KERNELS_GPU_CUSTOM_FUSED_BATCH_NORM_OP_H_

#include <string>
#include <unordered_map>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/common/fused_batch_norm_functor.h"
#include "itex/core/kernels/common/fused_batch_norm_op.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace functor {

template <typename Device, typename T, typename U, bool IsTrain>
struct DPCPPFusedBatchNorm {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  const Tensor& estimated_mean_input,
                  const Tensor& estimated_variance_input,
                  const Tensor* side_input_tensor, U epsilon,
                  U exponential_avg_factor, Tensor* y_output,
                  Tensor* running_mean_output, Tensor* running_var_output,
                  Tensor* saved_mean_output, Tensor* saved_var_output,
                  TensorFormat tensor_format, bool use_reserved_space,
                  bool fuse_norm_relu, bool fuse_norm_add_relu, const int sp,
                  const int ic) {
    // assum tensor format is NHWC
    auto dev = context->eigen_gpu_device();

    Tensor transformed_x;
    Tensor transformed_y;
    constexpr int NDIMS = 3;
    const int in_batch = GetTensorDim(x_input, tensor_format, 'N');
    const int in_spatial = sp / in_batch;

    const T* x = x_input.flat<T>().data();
    T* y = y_output->flat<T>().data();

    if (tensor_format == FORMAT_NCHW) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  TensorShape({in_batch, in_spatial, ic}),
                                  &transformed_x));
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  TensorShape({in_batch, in_spatial, ic}),
                                  &transformed_y));
      // Perform NCHW to NHWC
      Eigen::array<int, NDIMS> perm = {0, 2, 1};
      auto x_ = typename TTypes<T, NDIMS>::ConstTensor(
          reinterpret_cast<const T*>(x_input.tensor_data().data()),
          {in_batch, ic, in_spatial});
      auto trans_x_ = typename TTypes<T, NDIMS>::Tensor(
          reinterpret_cast<T*>(
              const_cast<char*>(transformed_x.tensor_data().data())),
          {in_batch, in_spatial, ic});
      trans_x_.device(dev) = x_.shuffle(perm);
      x = transformed_x.flat<T>().data();
      y = transformed_y.flat<T>().data();
    }

    const U* scale = scale_input.flat<U>().data();
    const U* offset = offset_input.flat<U>().data();
    const U* old_mean = nullptr;
    const U* old_var = nullptr;
    if (!IsTrain || (IsTrain && exponential_avg_factor != 1)) {
      old_mean = estimated_mean_input.flat<U>().data();
      old_var = estimated_variance_input.flat<U>().data();
    }

    U* new_mean = running_mean_output->flat<U>().data();
    U* new_var = running_var_output->flat<U>().data();
    U* saved_mean = saved_mean_output->flat<U>().data();
    U* saved_var = saved_var_output->flat<U>().data();
    const T* side_input =
        fuse_norm_add_relu ? side_input_tensor->flat<T>().data() : nullptr;

    if (IsTrain) {
      Tensor temp_tensor;
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<U>::value,
                                          TensorShape({ic * 2}), &temp_tensor));
      U* temp_mean = temp_tensor.flat<U>().data();
      U* temp_var = temp_mean + ic;
      MeanVarReduction(context, x, temp_mean, temp_var, 1, sp, ic);

      if (fuse_norm_relu)
        BNForward<T, U, true, true, false>(
            context, x, temp_mean, temp_var, scale, offset, side_input, y,
            old_mean, old_var, new_mean, new_var, saved_mean, saved_var, sp, ic,
            epsilon, exponential_avg_factor);
      else if (fuse_norm_add_relu)
        BNForward<T, U, true, false, true>(
            context, x, temp_mean, temp_var, scale, offset, side_input, y,
            old_mean, old_var, new_mean, new_var, saved_mean, saved_var, sp, ic,
            epsilon, exponential_avg_factor);
      else
        BNForward<T, U, true, false, false>(
            context, x, temp_mean, temp_var, scale, offset, side_input, y,
            old_mean, old_var, new_mean, new_var, saved_mean, saved_var, sp, ic,
            epsilon, exponential_avg_factor);
    } else {
      if (fuse_norm_relu)
        BNForward<T, U, false, true, false>(
            context, x, const_cast<U*>(old_mean), const_cast<U*>(old_var),
            scale, offset, side_input, y, old_mean, old_var, new_mean, new_var,
            saved_mean, saved_var, sp, ic, epsilon, exponential_avg_factor);
      else if (fuse_norm_add_relu)
        BNForward<T, U, false, false, true>(
            context, x, const_cast<U*>(old_mean), const_cast<U*>(old_var),
            scale, offset, side_input, y, old_mean, old_var, new_mean, new_var,
            saved_mean, saved_var, sp, ic, epsilon, exponential_avg_factor);
      else
        BNForward<T, U, false, false, false>(
            context, x, const_cast<U*>(old_mean), const_cast<U*>(old_var),
            scale, offset, side_input, y, old_mean, old_var, new_mean, new_var,
            saved_mean, saved_var, sp, ic, epsilon, exponential_avg_factor);
    }

    if (tensor_format == FORMAT_NCHW) {
      // Perform NHWC to NCHW
      Eigen::array<int, NDIMS> perm = {0, 2, 1};
      auto trans_y_ = typename TTypes<T, NDIMS>::ConstTensor(
          reinterpret_cast<const T*>(transformed_y.tensor_data().data()),
          {in_batch, in_spatial, ic});
      auto y_ = typename TTypes<T, NDIMS>::Tensor(
          reinterpret_cast<T*>(
              const_cast<char*>(y_output->tensor_data().data())),
          {in_batch, ic, in_spatial});
      y_.device(dev) = trans_y_.shuffle(perm);
    }
  }
};

template <typename Device, typename T, typename U>
struct DPCPPFusedBatchNormGrad {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& mean_input, const Tensor& var_input,
                  const Tensor* y_tensor, U epsilon, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output, Tensor* offset_backprop_output,
                  Tensor* diff_side_input_backprop_output,
                  TensorFormat tensor_format, bool fuse_norm_relu,
                  bool fuse_norm_add_relu, const int sp, const int ic) {
    auto dev = context->eigen_gpu_device();

    const T* x = x_input.flat<T>().data();
    const T* dy = y_backprop_input.flat<T>().data();
    const T* y = (fuse_norm_relu || fuse_norm_add_relu)
                     ? y_tensor->flat<T>().data()
                     : nullptr;
    T* dx = x_backprop_output->flat<T>().data();
    T* dside_x = fuse_norm_add_relu
                     ? diff_side_input_backprop_output->flat<T>().data()
                     : nullptr;

    constexpr int NDIMS = 3;
    Eigen::array<int, NDIMS> perm = {0, 2, 1};
    const int in_batch = GetTensorDim(x_input, tensor_format, 'N');
    const int in_spatial = sp / in_batch;

    Tensor transformed_x_input;
    Tensor transformed_y_backprop_input;
    Tensor transformed_x_backprop_output;
    Tensor transformed_y_input;
    Tensor transformed_side_x_backprop_output;
    if (tensor_format == FORMAT_NCHW) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  TensorShape({in_batch, in_spatial, ic}),
                                  &transformed_x_input));
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  TensorShape({in_batch, in_spatial, ic}),
                                  &transformed_y_backprop_input));
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  TensorShape({in_batch, in_spatial, ic}),
                                  &transformed_x_backprop_output));
      if (fuse_norm_relu || fuse_norm_add_relu) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value,
                                    TensorShape({in_batch, in_spatial, ic}),
                                    &transformed_y_input));
        auto y_ = typename TTypes<T, NDIMS>::ConstTensor(
            reinterpret_cast<const T*>(y_tensor->tensor_data().data()),
            {in_batch, ic, in_spatial});
        auto trans_y_ = typename TTypes<T, NDIMS>::Tensor(
            reinterpret_cast<T*>(
                const_cast<char*>(transformed_y_input.tensor_data().data())),
            {in_batch, in_spatial, ic});
        trans_y_.device(dev) = y_.shuffle(perm);
        y = transformed_y_input.flat<T>().data();
      }

      if (fuse_norm_add_relu) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value,
                                    TensorShape({in_batch, in_spatial, ic}),
                                    &transformed_side_x_backprop_output));
        dside_x = transformed_side_x_backprop_output.flat<T>().data();
      }
      // Perform NCHW to NHWC
      auto x_ = typename TTypes<T, NDIMS>::ConstTensor(
          reinterpret_cast<const T*>(x_input.tensor_data().data()),
          {in_batch, ic, in_spatial});
      auto trans_x_ = typename TTypes<T, NDIMS>::Tensor(
          reinterpret_cast<T*>(
              const_cast<char*>(transformed_x_input.tensor_data().data())),
          {in_batch, in_spatial, ic});
      trans_x_.device(dev) = x_.shuffle(perm);

      auto dy_ = typename TTypes<T, NDIMS>::ConstTensor(
          reinterpret_cast<const T*>(y_backprop_input.tensor_data().data()),
          {in_batch, ic, in_spatial});
      auto trans_dy_ = typename TTypes<T, NDIMS>::Tensor(
          reinterpret_cast<T*>(const_cast<char*>(
              transformed_y_backprop_input.tensor_data().data())),
          {in_batch, in_spatial, ic});
      trans_dy_.device(dev) = dy_.shuffle(perm);

      x = transformed_x_input.flat<T>().data();
      dy = transformed_y_backprop_input.flat<T>().data();
      dx = transformed_x_backprop_output.flat<T>().data();
    }

    const U* mean = mean_input.flat<U>().data();
    const U* var = var_input.flat<U>().data();
    const U* scale = scale_input.flat<U>().data();

    U* dscale = scale_backprop_output->flat<U>().data();
    U* doffset = offset_backprop_output->flat<U>().data();

    Tensor sum_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          TensorShape({2 * ic}), &sum_tensor));
    U* sum_dy = sum_tensor.flat<U>().data();
    U* sum_dy_x_center = sum_dy + ic;

    if (fuse_norm_relu) {
      BnormBwkReduction<T, U, true, false>(context, x, dy, mean, y, sum_dy,
                                           sum_dy_x_center, 1, sp, ic);
      BnBackward<T, U, true, false>(context, x, dy, mean, var, y, scale, sum_dy,
                                    sum_dy_x_center, dx, dscale, doffset,
                                    dside_x, sp, ic, epsilon);
    } else if (fuse_norm_add_relu) {
      BnormBwkReduction<T, U, false, true>(context, x, dy, mean, y, sum_dy,
                                           sum_dy_x_center, 1, sp, ic);
      BnBackward<T, U, false, true>(context, x, dy, mean, var, y, scale, sum_dy,
                                    sum_dy_x_center, dx, dscale, doffset,
                                    dside_x, sp, ic, epsilon);
    } else {
      BnormBwkReduction<T, U, false, false>(context, x, dy, mean, y, sum_dy,
                                            sum_dy_x_center, 1, sp, ic);
      BnBackward<T, U, false, false>(context, x, dy, mean, var, y, scale,
                                     sum_dy, sum_dy_x_center, dx, dscale,
                                     doffset, dside_x, sp, ic, epsilon);
    }

    if (tensor_format == FORMAT_NCHW) {
      // Perform NHWC to NCHW
      auto trans_dx_ = typename TTypes<T, NDIMS>::ConstTensor(
          reinterpret_cast<const T*>(
              transformed_x_backprop_output.tensor_data().data()),
          {in_batch, in_spatial, ic});
      auto dx_ = typename TTypes<T, NDIMS>::Tensor(
          reinterpret_cast<T*>(
              const_cast<char*>(x_backprop_output->tensor_data().data())),
          {in_batch, ic, in_spatial});
      dx_.device(dev) = trans_dx_.shuffle(perm);

      if (fuse_norm_add_relu) {
        auto trans_dside_x_ = typename TTypes<T, NDIMS>::ConstTensor(
            reinterpret_cast<const T*>(
                transformed_side_x_backprop_output.tensor_data().data()),
            {in_batch, in_spatial, ic});
        auto dside_x_ = typename TTypes<T, NDIMS>::Tensor(
            reinterpret_cast<T*>(const_cast<char*>(
                diff_side_input_backprop_output->tensor_data().data())),
            {in_batch, ic, in_spatial});
        dside_x_.device(dev) = trans_dside_x_.shuffle(perm);
      }
    }
  }
};

}  // namespace functor

template <typename Device, typename T, typename U, bool UseReservedSpace,
          bool IsBatchNormEx>
class CustomFusedBatchNormOp
    : public FusedBatchNormOp<Device, T, U, UseReservedSpace, IsBatchNormEx> {
  static constexpr bool use_reserved_space = UseReservedSpace;

 public:
  explicit CustomFusedBatchNormOp(OpKernelConstruction* context)
      : FusedBatchNormOp<Device, T, U, UseReservedSpace, IsBatchNormEx>(
            context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    float exponential_avg_factor;
    OP_REQUIRES_OK(context, context->GetAttr("exponential_avg_factor",
                                             &exponential_avg_factor));
    exponential_avg_factor_ = static_cast<U>(exponential_avg_factor);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (IsBatchNormEx) {
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(context, num_side_inputs >= 0 && num_side_inputs <= 1,
                  errors::InvalidArgument(
                      "FusedBatchNorm accepts at most one side input."));
      has_side_input_ = (num_side_inputs == 1);

      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode_));
      OP_REQUIRES(context, activation_mode_ == FbnActivationMode::kRelu,
                  errors::InvalidArgument(
                      "FusedBatchNorm only support Relu activation"));
    } else {
      has_side_input_ = false;
      activation_mode_ = FbnActivationMode::kIdentity;
    }
  }
  // Note: There is two fusion for batchnorm
  // BN + relu, no side input, no workspace need
  // BN + add + relu, the other input of add work as side input, no workspace
  // need
  void Compute(OpKernelContext* context) override {
    // fall back to onednn impl if format=NCHW
    if (tensor_format_ == FORMAT_NCHW) {
      FusedBatchNormOp<Device, T, U, UseReservedSpace, IsBatchNormEx>::Compute(
          context);
      return;
    }

    Tensor x = context->input(0);
    const Tensor& scale = context->input(1);
    const Tensor& offset = context->input(2);
    const Tensor& estimated_mean = context->input(3);
    const Tensor& estimated_var = context->input(4);
    const Tensor* side_input = has_side_input_ ? &context->input(5) : nullptr;

    OP_REQUIRES(
        context, x.dims() == 4 || x.dims() == 5,
        errors::InvalidArgument("input must be 4-dimensional or 5-dimensional",
                                x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    OP_REQUIRES(context, estimated_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        estimated_mean.shape().DebugString()));
    OP_REQUIRES(
        context, estimated_var.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                estimated_var.shape().DebugString()));
    // Note, For BN + Add + Relu fusion, BN is one input of Add,
    // and has same shape with the other input
    if (has_side_input_) {
      OP_REQUIRES(context, side_input->shape() == x.shape(),
                  errors::InvalidArgument(
                      "side_input shape must be equal to input shape: ",
                      side_input->shape().DebugString(),
                      " != ", x.shape().DebugString()));
    }

    // Allocate 5 output TF tensors.
    Tensor* running_mean = nullptr;
    Tensor* running_var = nullptr;
    Tensor* saved_mean = nullptr;
    Tensor* saved_var = nullptr;
    Tensor* reserved_space = nullptr;

    // Handle the special case: input with 0 elements and 0 batch size.
    Tensor* y = nullptr;
    TensorShape x_shape = x.shape();
    TensorShape workspace_shape;
    if (x_shape.num_elements() == 0) {
      workspace_shape.AddDim(0);
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {0}, 0, x_shape, &y));
      ITEX_DCHECK(y);
      AllocateTFOutputs(context, scale.shape(), workspace_shape, &running_mean,
                        &running_var, &saved_mean, &saved_var, &reserved_space,
                        true);

      return;
    } else {
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {0}, 0, x_shape, &y));
    }

    // There is actually no workspace tensor out, so we make a dummy one.
    workspace_shape.AddDim(0);
    AllocateTFOutputs(context, scale.shape(), workspace_shape, &running_mean,
                      &running_var, &saved_mean, &saved_var, &reserved_space);

    int ic = static_cast<int>(GetTensorDim(x, tensor_format_, 'C'));
    int sp = x.NumElements() / ic;
    bool fuse_norm_relu =
        (!has_side_input_) && (activation_mode_ == FbnActivationMode::kRelu);
    bool fuse_norm_add_relu =
        (has_side_input_) && (activation_mode_ == FbnActivationMode::kRelu);
    if (is_training_) {
      functor::DPCPPFusedBatchNorm<Device, T, U, 1>()(
          context, x, scale, offset, estimated_mean, estimated_var, side_input,
          epsilon_, exponential_avg_factor_, y, running_mean, running_var,
          saved_mean, saved_var, tensor_format_, use_reserved_space,
          fuse_norm_relu, fuse_norm_add_relu, sp, ic);
    } else {
      functor::DPCPPFusedBatchNorm<Device, T, U, 0>()(
          context, x, scale, offset, estimated_mean, estimated_var, side_input,
          epsilon_, exponential_avg_factor_, y, running_mean, running_var,
          saved_mean, saved_var, tensor_format_, use_reserved_space,
          fuse_norm_relu, fuse_norm_add_relu, sp, ic);
    }
  }

 private:
  float epsilon_;
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_;
  FbnActivationMode activation_mode_;

  void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                         TensorShape workspace_tf_shape, Tensor** running_mean,
                         Tensor** running_var, Tensor** saved_mean,
                         Tensor** saved_var, Tensor** reserved_space,
                         bool init_val = false) {
    ITEX_DCHECK(running_mean);
    ITEX_DCHECK(running_var);
    ITEX_DCHECK(saved_mean);
    ITEX_DCHECK(saved_var);

    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {3}, 1, tf_shape_scale, running_mean));
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {4}, 2, tf_shape_scale, running_var));
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, tf_shape_scale, saved_mean));
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, tf_shape_scale, saved_var));

    if (init_val) {
      U nan = Eigen::NumTraits<U>::quiet_NaN();
      auto* stream = context->GetDeviceStream();
      const int kSize = tf_shape_scale.num_elements();
      DeviceFill<Device, U>((*running_mean)->flat<U>().data(), nan, kSize,
                            stream);
      DeviceFill<Device, U>((*running_var)->flat<U>().data(), nan, kSize,
                            stream);
      DeviceFill<Device, U>((*saved_mean)->flat<U>().data(), U(0), kSize,
                            stream);
      DeviceFill<Device, U>((*saved_var)->flat<U>().data(), U(0), kSize,
                            stream);
    }

    if (use_reserved_space)
      OP_REQUIRES_OK(context, context->allocate_output(5, workspace_tf_shape,
                                                       reserved_space));
  }
};

template <typename Device, typename T, typename U, bool ReservedSpace,
          bool IsBatchNormEx = false>
class CustomFusedBatchNormGradOp
    : public FusedBatchNormGradOp<Device, T, U, ReservedSpace, IsBatchNormEx> {
 public:
  explicit CustomFusedBatchNormGradOp(OpKernelConstruction* context)
      : FusedBatchNormGradOp<Device, T, U, ReservedSpace, IsBatchNormEx>(
            context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (IsBatchNormEx) {
      FbnActivationMode activation_mode;
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      if (num_side_inputs > 0) has_side_input_ = true;
      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode));
      OP_REQUIRES(context, activation_mode == FbnActivationMode::kReluGrad,
                  errors::InvalidArgument(
                      "FusedBatchNormGrad only support ReluGrad activation"));
    }
  }

  void Compute(OpKernelContext* context) override {
    // fall back to onednn impl if format=NCHW
    if (tensor_format_ == FORMAT_NCHW) {
      FusedBatchNormGradOp<Device, T, U, ReservedSpace, IsBatchNormEx>::Compute(
          context);
      return;
    }
    const Tensor& diff_dst_tensor = context->input(0);
    const Tensor& src_tensor = context->input(1);
    const Tensor& scale_tensor = context->input(2);
    const Tensor& saved_mean = context->input(3);
    const Tensor& saved_var = context->input(4);
    const Tensor* y_tensor = IsBatchNormEx ? &context->input(7) : nullptr;

    TensorShape tf_shape_src = src_tensor.shape();
    TensorShape tf_shape_diff_dst = diff_dst_tensor.shape();

    OP_REQUIRES(
        context, diff_dst_tensor.dims() == 4 || diff_dst_tensor.dims() == 5,
        errors::InvalidArgument("input must be 4-dimensional or 5-dimensional",
                                diff_dst_tensor.shape().DebugString()));
    OP_REQUIRES(
        context, src_tensor.dims() == 4 || src_tensor.dims() == 5,
        errors::InvalidArgument("input must be 4-dimensional or 5-dimensional",
                                src_tensor.shape().DebugString()));
    OP_REQUIRES(context, scale_tensor.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale_tensor.shape().DebugString()));
    OP_REQUIRES(context, saved_mean.dims() == 1,
                errors::InvalidArgument("saved mean must be 1-dimensional",
                                        saved_mean.shape().DebugString()));
    OP_REQUIRES(context, saved_var.dims() == 1,
                errors::InvalidArgument("saved variance must be 1-dimensional",
                                        saved_var.shape().DebugString()));

    // Allocate output TF tensors diff_scale and diff_shift.
    Tensor* diff_scale_tensor = nullptr;
    Tensor* diff_offset_tensor = nullptr;

    // Handle the special case: input with 0 element and 0 batch size.
    const int kDiffSrcIndex = 0;  // index of diff_src tensor
    Tensor* diff_src_tensor = nullptr;
    Tensor* diff_side_input_tensor = nullptr;
    if (tf_shape_src.num_elements() == 0 ||
        tf_shape_diff_dst.num_elements() == 0) {
      OP_REQUIRES_OK(context,
                     context->allocate_output(kDiffSrcIndex, tf_shape_src,
                                              &diff_src_tensor));
      ITEX_DCHECK(diff_src_tensor);

      auto diff_src_data = diff_src_tensor->flat<T>().data();
      std::fill_n(diff_src_data, diff_src_tensor->shape().num_elements(),
                  static_cast<T>(0));
      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_offset_tensor, true);

      return;
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(kDiffSrcIndex, src_tensor.shape(),
                                              &diff_src_tensor));
      // Allocate output TF tensors diff_scale and diff_shift.
      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_offset_tensor);
      if (has_side_input_) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(5, src_tensor.shape(),
                                                &diff_side_input_tensor));
      }
    }

    int ic = GetTensorDim(src_tensor, tensor_format_, 'C');
    int sp = tf_shape_src.num_elements() / ic;

    bool fuse_norm_relu = IsBatchNormEx && !has_side_input_;
    bool fuse_norm_add_relu = IsBatchNormEx && has_side_input_;
    functor::DPCPPFusedBatchNormGrad<Device, T, U>()(
        context, diff_dst_tensor, src_tensor, scale_tensor, saved_mean,
        saved_var, y_tensor, epsilon_, diff_src_tensor, diff_scale_tensor,
        diff_offset_tensor, diff_side_input_tensor, tensor_format_,
        fuse_norm_relu, fuse_norm_add_relu, sp, ic);
  }

 private:
  float epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_ = false;

  void AllocateTFOutputs(OpKernelContext* context,
                         TensorShape tf_shape_scale_shift,
                         Tensor** diff_scale_tensor,
                         Tensor** diff_offset_tensor, bool init_val = false) {
    ITEX_DCHECK(diff_scale_tensor);
    ITEX_DCHECK(diff_offset_tensor);

    const int kDiffScaleIndex = 1;
    const int kDiffShiftIndex = 2;
    const int kP1Index = 3;
    const int kP2Index = 4;

    functor::SetZeroFunctor<Device, U> f_zero;

    // Separate out scale and shift grad and copy to individual tensors
    OP_REQUIRES_OK(
        context, context->allocate_output(kDiffScaleIndex, tf_shape_scale_shift,
                                          diff_scale_tensor));
    ITEX_DCHECK(*diff_scale_tensor);

    OP_REQUIRES_OK(
        context, context->allocate_output(kDiffShiftIndex, tf_shape_scale_shift,
                                          diff_offset_tensor));
    ITEX_DCHECK(*diff_offset_tensor);

    // Placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor *p1_tensor = nullptr, *p2_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(kP1Index, TensorShape({}),
                                                     &p1_tensor));
    ITEX_DCHECK(p1_tensor);
    OP_REQUIRES_OK(context, context->allocate_output(kP2Index, TensorShape({}),
                                                     &p2_tensor));
    ITEX_DCHECK(p2_tensor);

    if (init_val) {
      f_zero(context->eigen_device<Device>(), (*diff_scale_tensor)->flat<U>());
      f_zero(context->eigen_device<Device>(), (*diff_offset_tensor)->flat<U>());
      f_zero(context->eigen_device<Device>(), (p1_tensor)->flat<U>());
      f_zero(context->eigen_device<Device>(), (p2_tensor)->flat<U>());
    }
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_CUSTOM_FUSED_BATCH_NORM_OP_H_
