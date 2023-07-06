import tensorflow as tf
from intel_extension_for_tensorflow.python.fp8.autocast import (
  is_fp8_enabled,
  get_fp8_recipe,
  amax_and_scale_update,
  get_default_fp8_recipe,
  record_fp8_out_in_global_buffer,
  get_fp8_out_scale_inv,
)

class BaseModule:
  """Transformer base module."""
  def __init__(self, *args, **kwargs):
    super(BaseModule, self).__init__(*args, **kwargs)
    # fp8 related
    self.fp8 = False
    self.fp8_meta = {}
    self.fp8_meta["recipe"] = get_default_fp8_recipe()
    self.fp8_meta_tensors_initialized = False
  """
  Below are fp8 helper functions for transformer core op.
  We should allocate tensor for fp8 meta data, so please call fp8_init() at the first.
  
  pre_forward() / pre_backward() updates fp8 meta data before forward/backward excution.
  get_fp8_meta_tensors() will return a list of fp8 meta tensors, and should regard them as fp8 op inputs.
  post_forward() / post_backward() records fp8 output and its corresponding scale inv.

  Example for MLP (2-layer):
    self.fp8_init(num_fp8_inps=1, num_gemms=2, num_fp8_outs=1)
    @tf.custom_gradient
    def forward_func(x, y):
      self.pre_forward()
      fwd_fp8_meta_tensors = self.get_fp8_meta_tensors(x, y, max_fp8_outs=1, fwd=True)
      z = mlp_forward(x, y, fwd_fp8_meta_tensors)
      def grad_fn(upstream):
        self.pre_backward()
        bwd_fp8_meta_tensors = self.get_fp8_meta_tensors(upstream, max_fp8_outs=1, fwd=False)
        bwd_fp8_meta_tensors.append(fwd_fp8_meta_tensors[0])
        bwd_fp8_meta_tensors.append(fwd_fp8_meta_tensors[1])
        dx, dy = mlp_backward(x, y, bwd_fp8_meta_tensors)
        self.post_backward(dx, dy)
        return dx, dy
      return z, grad_fn
    z = forward_func(x, y)
    self.post_forward(z)
  """

  def set_fp8_meta_tensors(self, fwd):
    """Init scales and amaxes for fwd | bwd."""
    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
    fp8_meta = self.fp8_meta
    fp8_meta[fp8_meta_tensor_key] = {}

    num_fp8_gemm_tensors = (
      fp8_meta["num_gemms"] * 3 if fwd else fp8_meta["num_gemms"] * 2
    )
    fp8_meta[fp8_meta_tensor_key]["gemm"] = {}
    gemm_meta_tensors = fp8_meta[fp8_meta_tensor_key]["gemm"]
    if num_fp8_gemm_tensors > 0:
      gemm_meta_tensors["scale"] = tf.Variable(
        initial_value=tf.ones(num_fp8_gemm_tensors),
        trainable=False)
      gemm_meta_tensors["scale_inv"] = tf.Variable(
        initial_value=tf.ones(num_fp8_gemm_tensors),
        trainable=False)
      gemm_meta_tensors["amax_history"] = tf.Variable(
        initial_value=tf.zeros(
          [self.fp8_meta["recipe"].amax_history_len, num_fp8_gemm_tensors]),
        trainable=False)

    num_fp8_out_tensors = (
      fp8_meta["num_fp8_outs_fwd"] if fwd else fp8_meta["num_fp8_outs_bwd"]
    )
    for ind in range(0, num_fp8_out_tensors):
      fp8_meta[fp8_meta_tensor_key]["out" + str(ind)] = {}
      out_meta_tensors = fp8_meta[fp8_meta_tensor_key]["out" + str(ind)]
      out_meta_tensors["scale"] = tf.Variable(
        initial_value=tf.ones(1),
        trainable=False)
      out_meta_tensors["scale_inv"] = tf.Variable(
        initial_value=tf.ones(1),
        trainable=False)
      out_meta_tensors["amax_history"] = tf.Variable(
        initial_value=tf.zeros(
          [fp8_meta["recipe"].amax_history_len, 1]),
        trainable=False)

  def init_fp8_meta_tensors(self):
    """Init scales and amaxes."""
    if self.fp8_meta_tensors_initialized:
      return

    self.set_fp8_meta_tensors(fwd=True)
    self.set_fp8_meta_tensors(fwd=False)

  def record_fp8_out(self, *outs, fwd=True):
    """Record fp8 out and its scale inverse factor."""
    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
    fp8_index = 0

    for out in outs:
      if out.dtype == tf.int8:
        out_meta_tensors = (
          self.fp8_meta[fp8_meta_tensor_key]["out" + str(fp8_index)]
        )
        record_fp8_out_in_global_buffer(out, out_meta_tensors["scale_inv"])
        fp8_index = fp8_index + 1

  def update_fp8_meta_tensors(self, fwd=True):
    """Update fp8 meta tensor for gemms and fp8 outs."""
    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
    fp8_max_key = "fp8_max_fwd" if fwd else "fp8_max_bwd"

    fp8_meta = self.fp8_meta
    fp8_max = fp8_meta[fp8_max_key]
    recipe = fp8_meta["recipe"]

    num_fp8_out_tensors = (
      fp8_meta["num_fp8_outs_fwd"] if fwd else fp8_meta["num_fp8_outs_bwd"]
    )
    for ind in range(0, num_fp8_out_tensors):
      amax_and_scale_update(
        fp8_meta[fp8_meta_tensor_key]["out" + str(ind)],
        recipe, fp8_max)

    if fp8_meta["num_gemms"] > 0:
      amax_and_scale_update(
        fp8_meta[fp8_meta_tensor_key]["gemm"],
        recipe, fp8_max)

  """
  Currently, we only support fp8 as intermediate gemm in/out or transformer core op in/out.
  It is necessary to specify the number of gemms and fp8 inps/outs.
  """
  def fp8_init(self, num_fp8_inps=1, num_gemms=1, num_fp8_outs=1):
    """Initialize fp8 related metadata and tensors during fprop."""
    if not is_fp8_enabled():
      self.fp8 = False
      return
    # FP8 is already enabled and recipe is the same, don't do anything.
    if self.fp8 and get_fp8_recipe() == self.fp8_meta['recipe']:
      return

    # Set FP8, recipe, and other FP8 metadata.
    self.fp8 = True
    self.fp8_meta["recipe"] = get_fp8_recipe()
    self.fp8_meta["num_fp8_outs_fwd"] = num_fp8_outs
    self.fp8_meta["num_fp8_outs_bwd"] = num_fp8_inps
    self.fp8_meta["num_gemms"] = num_gemms

    # Set FP8_MAX per tensor according to recipe.
    fp8_format_val = self.fp8_meta["recipe"].fp8_format.value
    self.fp8_meta["fp8_max_fwd"] = fp8_format_val.max_fwd
    self.fp8_meta["fp8_max_bwd"] = fp8_format_val.max_bwd

    # Allocate scales and amaxes.
    self.init_fp8_meta_tensors()

  def get_fp8_meta_tensors(self, *inps, max_fp8_outs=1, fwd=True):
    "Get fp8 meta tensors for inputs, gemms and outputs."
    fp8_meta_tensors = [[], [], []]
    # fp8 meta tensor for inputs.
    for inp in inps:
      if inp.dtype == tf.int8:
        scale_inv = get_fp8_out_scale_inv(inp)
        fp8_meta_tensors[0].append(scale_inv)
      else:
        fp8_meta_tensors[0].append(tf.constant([], dtype=tf.float32))

    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

    # fp8 meta tensors for gemm.
    if self.fp8_meta["num_gemms"] > 0:
      fp8_meta_tensors[1].append(
        self.fp8_meta[fp8_meta_tensor_key]["gemm"]["amax_history"])
      fp8_meta_tensors[1].append(
        self.fp8_meta[fp8_meta_tensor_key]["gemm"]["scale"])
      fp8_meta_tensors[1].append(
        self.fp8_meta[fp8_meta_tensor_key]["gemm"]["scale_inv"])

    # fp8 meta tensor for outputs.
    num_fp8_out_tensors = (
      self.fp8_meta["num_fp8_outs_fwd"] if fwd \
      else self.fp8_meta["num_fp8_outs_bwd"]
    )
    for ind in range(0, max(num_fp8_out_tensors, max_fp8_outs)):
      if ind < num_fp8_out_tensors:
        fp8_meta_tensors[2].append(
          self.fp8_meta[fp8_meta_tensor_key]["out" + str(ind)]["amax_history"])
        fp8_meta_tensors[2].append(
          self.fp8_meta[fp8_meta_tensor_key]["out" + str(ind)]["scale"])
      else:
        fp8_meta_tensors[2].append(tf.constant([], dtype=tf.float32))
        fp8_meta_tensors[2].append(tf.constant([], dtype=tf.float32))
    return fp8_meta_tensors

  def pre_forward(self, training):
    """Update fp8 meta data before forward."""
    if self.fp8 and training:
      self.update_fp8_meta_tensors(fwd=True)

  def post_forward(self, *outs):
    """Record forward fp8 outputs."""
    self.record_fp8_out(*outs)

  def pre_backward(self):
    """Update fp8 meta data before backward."""
    self.update_fp8_meta_tensors(fwd=False)

  def post_backward(self, *outs):
    """Record backward fp8 outputs."""
    self.record_fp8_out(*outs, fwd=False)
