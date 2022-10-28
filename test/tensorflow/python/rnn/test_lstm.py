# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend
from tensorflow.python.ops import array_ops

from tensorflow.python.training import gradient_descent
import intel_extension_for_tensorflow as itex
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
import numpy as np
from tensorflow.keras.layers import (
    Input,
    Dense, Activation, Dropout, Lambda,
    Reshape, Flatten, Permute,  # RepeatVector
    SimpleRNN, BatchNormalization,
    Convolution1D, MaxPooling1D, TimeDistributed,
    Concatenate
    )
import time
import itertools
from tensorflow.keras import layers

Gates = 4
rtol = 1e-2
atol = 1e-2
np.random.seed(0)
tf.random.set_seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)
    
def bias_add(x, y):
    return x + y


def gemm(x, y):
    assert x.shape[-1] == y.shape[-2]
    return np.matmul(x, y)


def verify_backward_result(expected_gradients, gradients):
    np.testing.assert_allclose(
        expected_gradients["dx"], gradients["dx"], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        expected_gradients["dh"], gradients["dh"], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        expected_gradients["dc"], gradients["dc"], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        expected_gradients["dbias"], gradients["dbias"], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        expected_gradients["dwi"], gradients["dwi"], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        expected_gradients["dwh"], gradients["dwh"], rtol=rtol, atol=atol
    )
    print("---------------backward pass----------------------")


def verify_forward_result(outputs, hn, cn, outputs_, hn_, cn_):
    np.testing.assert_allclose(outputs, outputs_, rtol=rtol, atol=atol)
    np.testing.assert_allclose(hn, hn_, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cn, cn_, rtol=rtol, atol=atol)
    print("---------------forward pass----------------------")


def cell_func_forward(
    x, h_prev, c_prev, wei_ih, wei_hh, bias, input_dropout_mask, recurrent_dropout_mask
):
    x_dropout = x * input_dropout_mask
    wei_ii, wei_if, wei_ig, wei_io = np.split(wei_ih, Gates, axis=0)
    bias_i, bias_f, bias_g, bias_o = np.split(bias, Gates, axis=0)

    x_i = gemm(x_dropout[0], wei_ii.T)
    x_f = gemm(x_dropout[1], wei_if.T)
    x_g = gemm(x_dropout[2], wei_ig.T)
    x_o = gemm(x_dropout[3], wei_io.T)

    h_prev_dropout = h_prev * recurrent_dropout_mask
    wei_hi, wei_hf, wei_hg, wei_ho = np.split(wei_hh, Gates, axis=0)
    h_i = gemm(h_prev_dropout[0], wei_hi.T)
    h_f = gemm(h_prev_dropout[1], wei_hf.T)
    h_g = gemm(h_prev_dropout[2], wei_hg.T)
    h_o = gemm(h_prev_dropout[3], wei_ho.T)

    i = sigmoid(x_i + h_i + bias_i)
    f = sigmoid(x_f + h_f + bias_f)
    g = tanh(x_g + h_g + bias_g)
    o = sigmoid(x_o + h_o + bias_o)
    c_next = f * c_prev + g * i
    h_next = o * tanh(c_next)
    storage = [h_prev, c_prev, c_next, i, f, g, o]
    return (h_next, c_next), storage


# w_i(4 * hc, ic)
# w_h(4 * hc, hc)
def test_forward(
    x,
    h0,
    c0,
    w_i,
    w_h,
    bias,
    input_dropout_mask,
    recurrent_dropout_mask,
    seq_length,
    ic,
    uints,
    batch_size,
    require_grad=True,
):
    h_state = h0
    c_state = c0
    successive_outputs = []
    successive_states = []
    caches = []
    for step in range(seq_length):
        (h, c), cache = cell_func_forward(
            x[step],
            h_state,
            c_state,
            w_i,
            w_h,
            bias,
            input_dropout_mask,
            recurrent_dropout_mask,
        )
        caches.append(cache)
        h_state = h
        c_state = c
        successive_outputs.append(h)
        successive_states.append((h, c))
    successive_outputs = np.stack(successive_outputs)
    if require_grad:
        return successive_outputs, successive_states[-1], caches
    else:
        return successive_outputs, successive_states[-1]


def cell_func_backward(
    dst_diff_h,
    dst_diff_c,
    xt,
    workspace,
    w_i,
    w_h,
    input_dropout_mask,
    recurrent_dropout_mask,
):
    storage = workspace
    wei_hi, wei_hf, wei_hg, wei_ho = np.split(w_h, Gates, axis=0)
    wei_ii, wei_if, wei_ig, wei_io = np.split(w_i, Gates, axis=0)
    h_prev, c_prev, c_next, it, ft, gt, ot = storage

    xt_i = xt * input_dropout_mask[0]
    xt_f = xt * input_dropout_mask[1]
    xt_g = xt * input_dropout_mask[2]
    xt_o = xt * input_dropout_mask[3]
    h_prev_i = h_prev * recurrent_dropout_mask[0]
    h_prev_f = h_prev * recurrent_dropout_mask[1]
    h_prev_g = h_prev * recurrent_dropout_mask[2]
    h_prev_o = h_prev * recurrent_dropout_mask[3]

    dct = dst_diff_c + dst_diff_h * ot * (1 - np.square(tanh(c_next)))
    dc_prev = dct * ft

    dot = dst_diff_h * tanh(c_next) * ot * (1 - ot)
    dft = dct * c_prev * ft * (1 - ft)
    dit = dct * gt * it * (1 - it)
    dgt = dct * it * (1 - np.square(gt))

    dgates = np.concatenate((dit, dft, dgt, dot), axis=1)
    dbias = np.sum(dgates, axis=0)

    dw_io = gemm(xt_o.T, dot)
    dw_ho = gemm(h_prev_o.T, dot)

    dw_if = gemm(xt_f.T, dft)
    dw_hf = gemm(h_prev_f.T, dft)

    dw_ii = gemm(xt_i.T, dit)
    dw_hi = gemm(h_prev_i.T, dit)

    dw_ig = gemm(xt_g.T, dgt)
    dw_hg = gemm(h_prev_g.T, dgt)

    dw_i = np.concatenate((dw_ii, dw_if, dw_ig, dw_io), axis=1)
    dw_h = np.concatenate((dw_hi, dw_hf, dw_hg, dw_ho), axis=1)

    dh_prev_i = gemm(dit, wei_hi)
    dh_prev_f = gemm(dft, wei_hf)
    dh_prev_g = gemm(dgt, wei_hg)
    dh_prev_o = gemm(dot, wei_ho)
    dh_prev = (
        dh_prev_i * recurrent_dropout_mask[0]
        + dh_prev_f * recurrent_dropout_mask[1]
        + dh_prev_g * recurrent_dropout_mask[2]
        + dh_prev_o * recurrent_dropout_mask[3]
    )
    # dh_prev = gemm(dgates, w_h.T)
    # dxt = gemm(dgates, w_i.T)

    dxt_i = gemm(dit, wei_ii)
    dxt_f = gemm(dft, wei_if)
    dxt_g = gemm(dgt, wei_ig)
    dxt_o = gemm(dot, wei_io)
    dxt = (
        dxt_i * input_dropout_mask[0]
        + dxt_f * input_dropout_mask[1]
        + dxt_g * input_dropout_mask[2]
        + dxt_o * input_dropout_mask[3]
    )

    gradients = {
        "dx": dxt,
        "dh": dh_prev,
        "dc": dc_prev,
        "dwi": dw_i,
        "dwh": dw_h,
        "dbias": dbias,
    }
    return gradients


def test_backward(
    d_outputs,
    d_diff_h,
    d_diff_c,
    x,
    workspace,
    w_i,
    w_h,
    timesteps,
    ic,
    units,
    batch_size,
    input_dropout_mask,
    recurrent_dropout_mask,
):
    dw_h = np.zeros((units, Gates * units))
    dw_i = np.zeros((ic, Gates * units))
    dbias = np.zeros((Gates * units))
    dx = np.zeros((timesteps, batch_size, ic))

    dst_diff_h = np.zeros((batch_size, units))
    dst_diff_c = np.zeros((batch_size, units))
    dwi_list = []
    dwh_list = []
    dbias_list = []
    for step in reversed(range(timesteps)):
        dst_diff_h += d_outputs[step]
        gradients = cell_func_backward(
            dst_diff_h,
            dst_diff_c,
            x[step],
            workspace[step],
            w_i,
            w_h,
            input_dropout_mask,
            recurrent_dropout_mask,
        )
        dx[step] = gradients["dx"]
        dst_diff_h = gradients["dh"]
        dst_diff_c = gradients["dc"]
        dw_h += gradients["dwh"]
        dw_i += gradients["dwi"]
        dbias += gradients["dbias"]
        dwi_list.append(gradients["dwi"])
        dwh_list.append(gradients["dwh"])
        dbias_list.append(gradients["dbias"])
    return dict(dx=dx, dwh=dw_h, dwi=dw_i, dbias=dbias, dh=dst_diff_h, dc=dst_diff_c)


def store_result(**kw):
    for key in kw.keys():
        np.savetxt(
            "./data/" + key + "_array.txt", np.ravel(kw[key]), fmt="%.10f", newline=" "
        )


def transpose_weight(weight, transpose=False):
    array = np.split(weight, Gates, axis=1)
    transpose_array = [np.transpose(x) if transpose else x for x in array]
    return np.concatenate(transpose_array, axis=0)

# testing custom python code with keras layer
def test_custom_lstm_py_code(x_array, h_array, c_array, timesteps, ic, units, batch_size, dp, re_dp):
    print("------------testing custom python code with keras layer--------------")
    model_kwargs = dict(
        return_sequences=True,
        return_state=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        stateful=True,
        dropout=dp,
        recurrent_dropout=re_dp,
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
        time_major=True,
        implementation=1,
    )      
    x = tf.Variable(x_array, dtype=tf.float32)
    h0 = tf.Variable(h_array, trainable=True, dtype=tf.float32)
    c0 = tf.Variable(c_array, trainable=True, dtype=tf.float32)
    layer = tf.keras.layers.LSTM(units, **model_kwargs)

    with tf.GradientTape(persistent=True) as tape:
        outputs, hn, cn = layer(x, initial_state=[h0, c0], training=True)
        loss = tf.reduce_sum(outputs * 2)


    if model_kwargs["dropout"] > 0 and model_kwargs["dropout"] < 1:
        input_dropout_mask = layer.cell.get_dropout_mask_for_cell(
            x[0], training=True, count=Gates
        )
        input_dropout_mask = np.array([array.numpy() for array in input_dropout_mask])
    else:
        input_dropout_mask = np.array([np.ones((batch_size, ic)) for _ in range(Gates)])

    if model_kwargs["recurrent_dropout"] > 0 and model_kwargs["recurrent_dropout"] < 1:
        recurrent_dropout_mask = layer.cell.get_recurrent_dropout_mask_for_cell(
            h_array, training=True, count=Gates
        )
        recurrent_dropout_mask = np.array(
            [array.numpy() for array in recurrent_dropout_mask]
        )
    else:
        recurrent_dropout_mask = np.array(
            [np.ones((batch_size, units)) for _ in range(Gates)]
        )   
    doutputs = tape.gradient(loss, outputs)
    dhn = tape.gradient(loss, hn)
    dcn = tape.gradient(loss, cn)
    dh = tape.gradient(loss, h0)
    dc = tape.gradient(loss, c0)
    dx = tape.gradient(loss, x)
    dweights = tape.gradient(loss, layer.trainable_variables)
    gradients = dict(
        dx=dx, dh=dh, dc=dc, dwi=dweights[0], dwh=dweights[1], dbias=dweights[2]
    )

    # transpose weight
    wei_ih = transpose_weight(layer.trainable_variables[0], transpose=True)
    wei_hh = transpose_weight(layer.trainable_variables[1], transpose=True)
    bias = layer.trainable_variables[2]

    # forward test
    outputs_, (hn_, cn_), workspace = test_forward(
        x_array,
        h_array,
        c_array,
        wei_ih,  # kernel
        wei_hh,  # recurrent_kernel
        bias,
        input_dropout_mask,
        recurrent_dropout_mask,
        timesteps,
        ic,
        units,
        batch_size,
        require_grad=True,
    )

    verify_forward_result(outputs, hn, cn, outputs_, hn_, cn_)

    gradients_ = test_backward(
        doutputs,
        dhn,
        dcn,
        x_array,
        workspace,
        wei_ih,
        wei_hh,
        timesteps,
        ic,
        units,
        batch_size,
        input_dropout_mask,
        recurrent_dropout_mask,
    )

    verify_backward_result(gradients, gradients_)

# test ItexLSTM accuracy with custom python code
def test_itex_lstm(x_array, h_array, c_array, timesteps, ic, units, batch_size, dp, re_dp):
    model_kwargs = dict(
        return_sequences=True,
        return_state=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        stateful=True,
        dropout=dp,
        recurrent_dropout=re_dp,
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
        time_major=True,
    )        
    print("-----------------testing itex LSTM with custom python code------------------")    
    x = tf.Variable(x_array, dtype=tf.float32)
    h0 = tf.Variable(h_array, trainable=True, dtype=tf.float32)
    c0 = tf.Variable(c_array, trainable=True, dtype=tf.float32)
    layer = itex.ops.ItexLSTM(units, **model_kwargs)

    with tf.GradientTape(persistent=True) as tape:
        outputs, hn, cn = layer(x, initial_state=[h0, c0], training=True)
        loss = tf.reduce_sum(outputs * 2)
    doutputs = tape.gradient(loss, outputs)
    dhn = tape.gradient(loss, hn)
    dcn = tape.gradient(loss, cn)        
    dh = tape.gradient(loss, h0)
    dc = tape.gradient(loss, c0)
    dx = tape.gradient(loss, x)
    dweights = tape.gradient(loss, layer.trainable_variables)
    gradients = dict(
        dx=dx, dh=dh, dc=dc, dwi=dweights[0], dwh=dweights[1], dbias=dweights[2]
    )

    
    if model_kwargs["dropout"] > 0 and model_kwargs["dropout"] < 1:
        input_dropout_mask = layer.cell.get_dropout_mask_for_cell(
            x[0], training=True, count=Gates
        )
        input_dropout_mask = np.array([array.numpy() for array in input_dropout_mask])
    else:
        input_dropout_mask = np.array([np.ones((batch_size, ic)) for _ in range(Gates)])

    if model_kwargs["recurrent_dropout"] > 0 and model_kwargs["recurrent_dropout"] < 1:
        recurrent_dropout_mask = layer.cell.get_recurrent_dropout_mask_for_cell(
            h_array, training=True, count=Gates
        )
        recurrent_dropout_mask = np.array(
            [array.numpy() for array in recurrent_dropout_mask]
        )
    else:
        recurrent_dropout_mask = np.array(
            [np.ones((batch_size, units)) for _ in range(Gates)]
        )    
    
    # split and transpose weight
    wei_ih = transpose_weight(layer.trainable_variables[0], transpose=True)
    wei_hh = transpose_weight(layer.trainable_variables[1], transpose=True)
    bias = layer.trainable_variables[2]

    # forward test
    outputs_, (hn_, cn_), workspace = test_forward(
        x_array,
        h_array,
        c_array,
        wei_ih,  # kernel
        wei_hh,  # recurrent_kernel
        bias,
        input_dropout_mask,
        recurrent_dropout_mask,
        timesteps,
        ic,
        units,
        batch_size,
        require_grad=True,
    )
    verify_forward_result(outputs, hn, cn, outputs_, hn_, cn_)
    
    gradients_ = test_backward(
        doutputs,
        dhn,
        dcn,
        x_array,
        workspace,
        wei_ih,
        wei_hh,
        timesteps,
        ic,
        units,
        batch_size,
        input_dropout_mask,
        recurrent_dropout_mask,
    )
    
    verify_backward_result(gradients_, gradients)
    

def test():
    timesteps = 3
    ic = 4
    units = 5
    batch_size = 6

    x_array = np.random.randn(timesteps, batch_size, ic).astype(np.float32)
    h_array = np.random.randn(batch_size, units).astype(np.float32)
    c_array = np.random.randn(batch_size, units).astype(np.float32)
    
    # Skip case when dp=0.1 and re_dp=0, as tensorflow recurrent_v2 lstm use implementation 2 for this case
    # while itex lstm follow recurrent implementation 1 for this case
    for (dp, re_dp) in ([0, 0], [0, 0.1], [0.1, 0.1]):
        test_custom_lstm_py_code(x_array, h_array, c_array, timesteps, ic, units, batch_size, dp, re_dp)
        test_itex_lstm(x_array, h_array, c_array, timesteps, ic, units, batch_size, dp, re_dp)
    

if __name__ == "__main__":
    test()

