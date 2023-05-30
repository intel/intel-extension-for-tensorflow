from contextlib import contextmanager
from typing import Optional

import tensorflow as tf
from intel_extension_for_tensorflow.python.fp8 import DelayedScaling, Format

_FP8_ENABLED = False
_FP8_RECIPE = None
_global_fp8_outs_buffer = {}

def record_fp8_out_in_global_buffer(fp8_tensor, fp8_meta_tensor):
  """Map each fp8 out to its scale inv factor."""
  tensor_key = fp8_tensor.ref()
  if not tensor_key in _global_fp8_outs_buffer:
    _global_fp8_outs_buffer[tensor_key] = fp8_meta_tensor

def get_fp8_out_scale_inv(fp8_tensor):
  """Get fp8 scale inv factor."""
  tensor_key = fp8_tensor.ref()
  assert tensor_key in _global_fp8_outs_buffer
  return _global_fp8_outs_buffer[tensor_key]

def get_default_fp8_recipe():
  """
  FP8 recipe if not provided by user
  Margin = 0, interval = 1, HYBRID
  """
  return DelayedScaling()

def get_fp8_dtype(fp8_recipe, fwd=True):
  """Get fp8 data type according to recipe and tensor"""
  assert fp8_recipe.fp8_format == Format.HYBRID
  return "E4M3" if fwd else "E5M2"

def is_fp8_enabled():
  """Is FP8 enabled"""
  return _FP8_ENABLED

def get_fp8_recipe():
  """Return the fp8 recipe"""
  return _FP8_RECIPE

# TODO(ITEX): Plan to decorate this func by tf.function(jit_compile=True).
def _default_sf_compute(amax, scale, fp8_max, margin):
  """Default function to convert amax to scaling factor."""
  exp = tf.math.floor(tf.experimental.numpy.log2(fp8_max / amax)) - margin
  sf = tf.math.round(tf.math.pow(float(2.0), tf.math.abs(exp)))
  sf = tf.where(amax > float(0.0), sf, scale)
  sf = tf.where(tf.math.is_finite(amax), sf, scale)
  sf = tf.where(exp < float(0.0), float(1.0) / sf, sf)
  return sf

def _roll_and_zero_out(amax_history):
  """Update amax history and set next amax to zero."""
  amax_history = tf.roll(amax_history, -1, 0)
  zeros = tf.zeros(shape=amax_history[0].shape)
  updated = tf.tensor_scatter_nd_update(amax_history, [[0]], [zeros])
  return updated

def _reduce_max_and_default_sf_compute(amax_history, scale, fp8_max, margin):
  """Get amax using max algorithm and compute scaling factor."""
  amax = tf.reduce_max(amax_history, axis=0)
  sf = _default_sf_compute(amax, scale, fp8_max, margin)
  updated = _roll_and_zero_out(amax_history)
  return updated, sf

def _most_recent_and_default_sf_compute(amax_history, scale, fp8_max, margin):
  """Get amax using most-recent algorithm and compute scaling factor."""
  amax = amax_history[0]
  sf = _default_sf_compute(amax, scale, fp8_max, margin)
  updated = _roll_and_zero_out(amax_history)
  return updated, sf

def fused_amax_and_scale_update(
    amax_history,
    scale,
    fp8_max,
    margin,
    amax_compute_algo,
):
  """Amax to scale conversion."""
  if amax_compute_algo == "max":
    updated, sf = _reduce_max_and_default_sf_compute(
      amax_history, scale, fp8_max, margin
    )
  else:
    assert amax_compute_algo == "most_recent"
    updated, sf = _most_recent_and_default_sf_compute(
      amax_history, scale, fp8_max, margin
    )

  return updated, sf

def amax_and_scale_update(fp8_meta_tensors, recipe, fp8_max):
  """Updates fp8 amaxes/scales for fwd | bwd."""
  amax_compute = recipe.amax_compute_algo
  sf_compute = recipe.scaling_factor_compute_algo

  if not callable(amax_compute) and sf_compute is None:
    amax, scale = fused_amax_and_scale_update(
      fp8_meta_tensors["amax_history"],
      fp8_meta_tensors["scale"],
      fp8_max,
      recipe.margin,
      recipe.amax_compute_algo,
    )
    fp8_meta_tensors["amax_history"] = (
      fp8_meta_tensors["amax_history"].assign(amax)
    )
    fp8_meta_tensors["scale"] = (
      fp8_meta_tensors["scale"].assign(scale)
    )
    fp8_meta_tensors["scale_inv"] = (
      fp8_meta_tensors["scale_inv"].assign(1.0 / scale)
    )
  else:
    raise ValueError(
      "We only support the fp8 recipe with 'max' or 'most_recent' "
      "amax_compute_algo and default scaling_factor_compute_algo at this "
      "moment."
    )

@contextmanager
def fp8_autocast(
  enabled: bool = False,
  fp8_recipe: Optional[DelayedScaling] = None
):
  """
  Context manager for FP8 usage.

  .. code-block:: python

    with fp8_autocast(enabled=True):
      out = model(inp)

  Parameters
  ----------
  enabled: bool, default = `False`
           whether or not to enable fp8
  fp8_recipe: recipe.DelayedScaling, default = `None`
              recipe used for FP8 training.
  """
  global _FP8_ENABLED, _FP8_RECIPE
  fp8_state = (_FP8_ENABLED, _FP8_RECIPE)
  try:
    _FP8_ENABLED = enabled
    _FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
    yield
  finally:
    _FP8_ENABLED, _FP8_RECIPE = fp8_state
