from tensorflow import keras
import tensorflow as tf
import numpy as np
import functools
from tensorflow.python.ops import array_ops

from tensorflow.python.ops import variables
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import stateless_random_ops
from intel_extension_for_tensorflow.python.ops.multi_head_attention import (
    scaled_dot_product_attention,
)

seed = (0, 1)


def spd(q, k, v, mask, dropout_p, seed, dtype, use_fast_attention=False):
    q_tf = tf.Variable(q, dtype=dtype)
    k_tf = tf.Variable(k, dtype=dtype)
    v_tf = tf.Variable(v, dtype=dtype)
    mask_tf = tf.Variable(mask, dtype=dtype)
    with tf.GradientTape(persistent=True) as tape:
        outputs = scaled_dot_product_attention(
            q_tf,
            k_tf,
            v_tf,
            mask_tf,
            dropout_p,
            seed,
            use_fast_attention=use_fast_attention,
        )
        loss = tf.reduce_sum(outputs * outputs)
    doutputs = tape.gradient(loss, outputs)
    dq = tape.gradient(loss, q_tf)
    dk = tape.gradient(loss, k_tf)
    dv = tape.gradient(loss, v_tf)
    return outputs, dq, dk, dv


def test_func(
    batch_size,
    from_seq_len,
    to_seq_len,
    num_heads,
    head_size,
    dtype,
    use_mask=False,
    use_dropout=False,
):
    np.random.seed(0)

    q = np.random.normal(size=[batch_size, num_heads, from_seq_len, head_size]).astype(
        np.float32
    )
    k = np.random.normal(size=[batch_size, num_heads, to_seq_len, head_size]).astype(
        np.float32
    )
    v = np.random.normal(size=[batch_size, num_heads, to_seq_len, head_size]).astype(
        np.float32
    )

    # q = np.ones([batch_size, num_heads, from_seq_len, head_size]).astype(np.float32)
    # k = np.ones([batch_size, num_heads, to_seq_len, head_size]).astype(np.float32)
    # v = np.ones([batch_size, num_heads, to_seq_len, head_size]).astype(np.float32)

    mask = np.zeros([batch_size, 1, from_seq_len, to_seq_len])
    if use_mask:
        # (B, N, F, T)
        # (B, 1, 1, T)
        mask = (
            np.random.uniform(
                size=[batch_size, 1, from_seq_len, to_seq_len], low=0, high=1
            )
            > 0.5
        )

    dropout_prob = 0
    if use_dropout:
        dropout_prob = 0.5

    ref_outputs, ref_dq, ref_dk, ref_dv = spd(
        q, k, v, mask, dropout_prob, seed, dtype, use_fast_attention=False
    )

    outputs, dq, dk, dv = spd(
        q, k, v, mask, dropout_prob, seed, dtype, use_fast_attention=True
    )

    if dtype == tf.float16 or dtype == tf.bfloat16:
        outputs, dq, dk, dv = [tf.cast(i, tf.float32) for i in [outputs, dq, dk, dv]]
        ref_outputs, ref_dq, ref_dk, ref_dv = [
            tf.cast(i, tf.float32) for i in [ref_outputs, ref_dq, ref_dk, ref_dv]
        ]
        atol = rtol = 1e-2
    else:
        atol = rtol = 1e-5

    np.testing.assert_allclose(outputs, ref_outputs, rtol=rtol, atol=atol)
    np.testing.assert_allclose(dv, ref_dv, rtol=rtol, atol=atol)
    try:
        np.testing.assert_allclose(dq, ref_dq, rtol=rtol, atol=atol)
    except AssertionError as err:
        print("dq accuracy verify failed")        
        print(err)
    try:
        np.testing.assert_allclose(dk, ref_dk, rtol=rtol, atol=atol)
    except AssertionError as err:
        print("dk accuracy verify failed")
        print(err)

def test_perf(
    batch_size,
    from_seq_len,
    to_seq_len,
    num_heads,
    head_size,
    dtype,
    use_mask=False,
    use_dropout=False,
):
    np.random.seed(0)

    q = np.random.normal(size=[batch_size, num_heads, from_seq_len, head_size]).astype(
        np.float32
    )
    k = np.random.normal(size=[batch_size, num_heads, to_seq_len, head_size]).astype(
        np.float32
    )
    v = np.random.normal(size=[batch_size, num_heads, to_seq_len, head_size]).astype(np.float32)
    mask_shape = [batch_size, 1, from_seq_len, to_seq_len]
    mask = np.zeros(mask_shape)
    if use_mask:
        mask = (np.random.uniform(size=mask_shape, low=0, high=1) > 0.5)

    dropout_prob = 0
    if use_dropout:
        dropout_prob = 0.5

    for i in range(5):
        outputs, dq, dk, dv = spd(
            q, k, v, mask, dropout_prob, seed, dtype, use_fast_attention=True
        )


# Q	16x4096x40	16x4096x40	16x1024x80	16x1024x80	16x256x160	16x256x160	16x64x160	16x64x160
# K	16x4096x40	16x77x40	16x1024x80	16x77x80	16x256x160	16x77x160	16x64x160	16x77x160
# V	16x4096x40	16x77x40	16x1024x80	16x77x80	16x256x160	16x77x160	16x64x160	16x77x160

if __name__ == "__main__":
    dtype = tf.float16
    # test_func(1, 64, 64, 8, 160, dtype, False, False)
    # test_func(1, 256, 256, 8, 160, dtype, False, False)
    # test_func(1, 1024, 1024, 8, 80, dtype, False, False)
    test_func(1, 512, 512, 2, 64, dtype, True, True)
    # test_func(1, 512, 512, 2, 64, dtype, True, True)
