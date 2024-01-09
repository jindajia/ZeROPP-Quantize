import sys,os
import torch
from zeropp.ops.op_builder.quantizer import CUDAQuantizer
from torch import linalg as LA
import math
def analysis_diff(origin_tensor, quantized_tensor):

    diff = origin_tensor - quantized_tensor
    abs_error_norm = LA.norm(diff)
    origin_norm = LA.norm(origin_tensor)
    rela_error_norm = abs_error_norm / origin_norm
    return rela_error_norm, abs_error_norm

def zeropp_quantization_test(input):
    """start quantization"""
    quantizer_module = CUDAQuantizer()
    groups = math.ceil(input.numel() / 2048)
    quantized_param, scales = quantizer_module.quantize(input, groups)
    print('quantized shape: {}, scales shape: {}'.format(quantized_param.shape, scales.shape))

    """dequantization"""
    """allocate buffer"""
    buffer_size = input.shape
    buffer_type = input.dtype
    param_buffer = torch.empty(
        buffer_size,
        dtype=buffer_type,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    """allocate buffer finished"""
    param_buffer.data = quantizer_module.dequantize(quantized_param, scales)
    return param_buffer

def get_normal_random_num(mean=0, std=1, size=(62237952, ), type=torch.float16, dev='cpu'):
    generate_tensor = torch.normal(mean=mean, std=std, size=size, dtype=type)
    if generate_tensor.device is not dev:
        generate_tensor.to(dev)
    return generate_tensor

def test_function():
    """load checkpoint tensor"""
    input = get_normal_random_num()
    print('tensor shape: {}, mean: {}, std: {}'.format(input.shape, torch.mean(input), torch.std(input)))

    zero_qt_data = zeropp_quantization_test(input)
    rela_error_norm, abs_error_norm = analysis_diff(zero_qt_data, input)
    print(f'abs error norm: {abs_error_norm}, relative error norm: {rela_error_norm}')

if __name__ == '__main__':
    test_function()