"""This module provides predefined FP8 recipes."""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Union, Callable, NamedTuple
from pydantic.dataclasses import dataclass


class _FormatHelper(NamedTuple):
  """
  Stores max FP8 values for fprop and bprop.
  """

  max_fwd: float
  max_bwd: float


class Format(Enum):
  """
  Supported FP8 formats.
  Values
  ------
  HYBRID :
          FP8 tensors in the forward pass are in e4m3 format,
          FP8 tensors in the backward pass are in e5m2 format
  """

  HYBRID = _FormatHelper(max_fwd=448, max_bwd=57344)

@dataclass()
class DelayedScaling:
  """
  Use the delayed scaling factor strategy.
  Use scale factor from previous iteration,
  and record amax history of `amax_history_len` steps.
  Parameters
  ----------
  margin : int, default = 0
           Margin for the scaling factor computation.
  fp8_format : {Format.HYBRID}, default = Format.HYBRID
               Controls the FP8 data format used during forward and backward
               pass.
  amax_history_len : int, default = 1024
                     The length of the amax history window used for
                     scaling factor computation.
  amax_compute_algo : {'max', 'most_recent', Callable}, default = 'max'
                      Algorithm used for choosing the `amax` value for the
                      scaling factor computation. There are 2 predefined
                      choices: `max` chooses the largest `amax` in the history
                      window, while `most_recent` always chooses the most
                      recently seen value. Alternatively, one may pass
                      a function of the signature:

                      .. code-block:: python

                        def amax_compute(amax_history: Tensor) -> Tensor

                      where `Tensor` is a framework tensor type.
  scaling_factor_compute_algo : Callable, default = None
                                Algorithm used for computing the new scaling
                                factor based on the value of `amax`. It should
                                be a function of the signature:

                                .. code-block:: python

                                  def scaling_factor_compute(
                                    amax: Tensor,
                                    old_scaling_factor: Tensor,
                                    fp8_max: Tensor,
                                    recipe: DelayedScaling) -> Tensor

                                where `Tensor` is a framework tensor type.

  Notes
  -----
  * By default (when `scaling_factor_compute_algo` is left as `None`)
    the scaling factor is computed from the final `amax` value
    using the formula:

    .. code-block:: python

        FP8_MAX = maximum_representable_value(fp8_format)
        exp = get_exponent(FP8_MAX / amax) - margin
        new_scaling_factor = 2.0 ^ exp

  * The scaling factor should always be a power of 2 to not introduce numerical
    error during the conversion from FP8 to higher precision format.
  """

  margin: int = 0
  fp8_format: Format = Format.HYBRID
  amax_history_len: int = 1024
  amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "max"
  scaling_factor_compute_algo: Optional[Callable] = None
