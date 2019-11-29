# -*- coding: utf-8 -*-


import math
import torch
        
        
class RepeatKernelConvFn(torch.autograd.function.Function):
    """2.5D convolution with kernel.
    """
        
    @staticmethod
    def forward(ctx, inputs, kernel, weight, bias=None, stride=1, padding=0, 
                dilation=1):

        (batch_size, channels), input_size = inputs.shape[:2], inputs.shape[2:]
        ctx.in_channels = channels
        ctx.input_size = input_size
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = torch.nn.modules.utils._pair(dilation)
        ctx.padding = torch.nn.modules.utils._pair(padding)
        ctx.stride = torch.nn.modules.utils._pair(stride)
        
        needs_input_grad = ctx.needs_input_grad
        ctx.save_for_backward(
            inputs if (needs_input_grad[1] or needs_input_grad[2]) else None,
            kernel if (needs_input_grad[0] or needs_input_grad[2]) else None,
            weight if (needs_input_grad[0] or needs_input_grad[1]) else None)
        ctx._backend = torch._thnn.type2backend[inputs.type()]
        

        inputs_wins = torch.nn.functional.unfold(inputs, ctx.kernel_size, 
                                                 ctx.dilation, ctx.padding,
                                                 ctx.stride)

        inputs_wins = inputs_wins.view(
            1, batch_size, channels, *kernel.shape[3:])
        inputs_mul_kernel = inputs_wins * kernel
                

        outputs = torch.einsum(
            'hijklmn,hojkl->iomn', (inputs_mul_kernel, weight))
        
        if bias is not None:
            outputs += bias.view(1, -1, 1, 1)
        return outputs
        
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_outputs):
        grad_inputs = grad_kernel = grad_weight = grad_bias = None
        batch_size, out_channels = grad_outputs.shape[:2]
        output_size = grad_outputs.shape[2:]
        in_channels = ctx.in_channels
        
        # Compute gradients
        inputs, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_inputs_mul_kernel = torch.einsum('iomn,hojkl->hijklmn',
                                                  (grad_outputs, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            inputs_wins = torch.nn.functional.unfold(inputs, ctx.kernel_size, 
                                                     ctx.dilation, ctx.padding,
                                                     ctx.stride)
            inputs_wins = inputs_wins.view(1, batch_size, in_channels,
                                           ctx.kernel_size[0], 
                                           ctx.kernel_size[1],
                                           output_size[0], output_size[1])
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.new()
            grad_inputs_wins = grad_inputs_mul_kernel * kernel
            grad_inputs_wins = grad_inputs_wins.view(
                ctx.kernel_size[0], batch_size, -1, output_size[0], output_size[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                                grad_inputs_wins,
                                                grad_inputs,
                                                ctx.input_size[0],
                                                ctx.input_size[1],
                                                ctx.kernel_size[0],
                                                ctx.kernel_size[1],
                                                ctx.dilation[0], 
                                                ctx.dilation[1],
                                                ctx.padding[0], 
                                                ctx.padding[1],
                                                ctx.stride[0],
                                                ctx.stride[1])
        if ctx.needs_input_grad[1]:
            grad_kernel = inputs_wins * grad_inputs_mul_kernel
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            inputs_mul_kernel = inputs_wins * kernel
            grad_weight = torch.einsum('iomn,hijklmn->hojkl',
                                       (grad_outputs, inputs_mul_kernel))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_outputs,))
        return (grad_inputs, grad_kernel, grad_weight, grad_bias, None, None,
                None)
        
        
class DepthKernelFn(torch.autograd.function.Function):
    """Compute mask in paper: 
        2.5D convolution for rgb-d semantic segmentation.
    """
    
    @staticmethod
    def forward(ctx, depth, f, kernel_size, stride, padding, dilation):

        ctx.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        ctx.stride = torch.nn.modules.utils._pair(stride)
        ctx.padding = torch.nn.modules.utils._pair(padding)
        ctx.dilation = torch.nn.modules.utils._pair(dilation)
        
        batch_size, channels, in_height, in_width = depth.shape
        out_height = (in_height + 2 * ctx.padding[0] - 
                      ctx.dilation[0] * (ctx.kernel_size[0] - 1)
                      -1) // ctx.stride[0] + 1
        out_width = (in_width + 2 * ctx.padding[1] - 
                     ctx.dilation[1] * (ctx.kernel_size[1] - 1)
                     -1) // ctx.stride[1] + 1
        
        depth_wins = torch.nn.functional.unfold(depth, ctx.kernel_size,
                                                ctx.dilation, ctx.padding,
                                                ctx.stride)
        depth_wins = depth_wins.view(batch_size, channels, ctx.kernel_size[0],
                                     ctx.kernel_size[1], out_height, out_width)
        s_i_wins = depth_wins/f
        
        kernels = []
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        for l in range(ctx.kernel_size[0]):
            z_l = depth_wins + (l - (ctx.kernel_size[0] - 1) / 2) * s_i_wins
            z_l_0 = z_l.contiguous()[:, :, center_y,
                    center_x, :, :]
            s_0 = s_i_wins.contiguous()[:, :, center_y,
                  center_x, :, :]
            a = z_l_0 - s_0 / 2
            b = z_l_0 + s_0 / 2
            a0 = torch.unsqueeze(a, dim=2)
            a1 = a0.repeat(1, 1, ctx.kernel_size[0]*ctx.kernel_size[1], 1, 1)
            a2 = a1.view(batch_size, channels, ctx.kernel_size[0],
                                     ctx.kernel_size[1], out_height, out_width)

            a3 = a2.transpose(2, 4).transpose(3, 5)

            b0 = torch.unsqueeze(b, dim=2)
            b1 = b0.repeat(1, 1, 9, 1, 1)
            b2 = b1.view(batch_size, channels, ctx.kernel_size[0],
                                     ctx.kernel_size[1], out_height, out_width)
            b3 = b2.transpose(2, 4).transpose(3, 5)
           
            mask_l_a3 = torch.where(depth_wins >= a2,
                                   torch.full_like(depth_wins, 1),
                                   torch.full_like(depth_wins, 0))
            mask_l_b3 = torch.where(depth_wins < b2,
                                   torch.full_like(depth_wins, 1),
                                   torch.full_like(depth_wins, 0))
            mask_l = torch.where(mask_l_a3 > mask_l_b3,
                                 torch.full_like(depth_wins, 0),
                                 mask_l_a3)
            kernels.append(mask_l.unsqueeze(dim=0))

        return torch.cat(kernels, dim=0)
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_outputs):
        return 0, None, None, None, None, None
    
    
class Conv2_5d(torch.nn.Module):
    """Implementation of 2.5D convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        """Constructor."""
        super(Conv2_5d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)
        
        # Parameters: weight, bias
        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(kernel_size, out_channels, in_channels, kernel_size,
                         kernel_size))
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialization
        self.reset_parameters()
        
    def forward(self, inputs, depth, f=1):

        kernel = DepthKernelFn.apply(depth, f, self.kernel_size, self.stride,
                                     self.padding, self.dilation)
        
        outputs = RepeatKernelConvFn.apply(inputs, kernel, self.weight,
                                           self.bias, self.stride,
                                           self.padding, self.dilation)
        return outputs
    

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
                
