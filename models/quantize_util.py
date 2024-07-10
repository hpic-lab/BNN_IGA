import torch
import torch.nn     as nn
from torch.autograd import Function

########################################   IGA   ########################################
def Binarize(tensor, quant_mode = 'det'): 
    if quant_mode == 'det':        
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def adc(PS, minimum = -128, maximum = 128, bit = 1):
    '''
    PS is patial sum and PSq is quantized patial sum
    this function define min values of Qin, Qout are same and also max value of Qin, Qout
    quantized level is defined as 2^bit - 1
    '''
    del_Qin  = (maximum - minimum) / (2**bit - 1)
    del_Qout = (maximum - minimum) / (2**bit - 1)
    PS_q = minimum + del_Qout*((PS + del_Qin/2 - minimum) // del_Qin)
    return PS_q

def gaussian_function(x, sigma=0.05):
    pi = 3.1415926535897934
    return torch.exp(-(x**2)/(2*(sigma**2))) / ((2*pi*(sigma**2))**0.5)

def impulse_gradient_approximation(x, minimum=-1, maximum=1, sigma=0.05, bit=1):
    '''
    this function is impulse_gradient_approximation (IGA) which is approximate gradient of quant function
    In this function, min values of Qin, Qout are same and also max value of Qin, Qout as adc function written above.
    '''
    del_Qin  = (maximum - minimum) / (2**bit - 1)
    del_Qout = (maximum - minimum) / (2**bit - 1)
    return del_Qout * gaussian_function(x=((x + del_Qin/2) % del_Qin) - del_Qin/2, sigma=sigma)

class WeightQuantF(Function):
    @staticmethod
    def forward(ctx, input, w_bit=1):
        ctx.save_for_backward(input)
        ctx.w_bit = w_bit
        output = input.new(input.size())
        if ctx.w_bit == 1: # binary
            output[input >  0] =  1
            output[input <= 0] = -1
        elif ctx.w_bit == 1.5: # ternary
            output[0.2<=input] = 1
            output[torch.logical_and(-0.2<input,input<0.2)] = 0
            output[input<=-0.2] = -1
        elif ctx.w_bit > 4: #full precision
            output = input
        else: # multibit
            output = adc(PS=input, minimum=-1, maximum=1, bit=ctx.w_bit)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if ctx.w_bit==1.5:
            quant_grad = gaussian_function(x=input-0.2,sigma=0.02) + gaussian_function(x=input+0.2,sigma=0.02)
        elif ctx.w_bit > 4:
            quatn_grad = 1
        else:
            quant_grad = impulse_gradient_approximation(x=input, minimum=-1, maximum=1, sigma=0.05, bit=ctx.w_bit)
        grad_input = grad_output.clone() * quant_grad
        return grad_input, None
QuantWeight = WeightQuantF.apply

class ActivationOutQuantF(Function):
    @staticmethod
    def forward(ctx, input, ao_bit=1):
        ctx.save_for_backward(input)
        ctx.ao_bit = ao_bit
        output = input.new(input.size())
        if ctx.ao_bit == 1: # binary
            output[input >  0] =  1
            output[input <= 0] = -1
        elif ctx.ao_bit == 1.5: # ternary
            output[0.2<=input] = 1
            output[torch.logical_and(-0.2<input,input<0.2)] = 0
            output[input<=-0.2] = -1
        elif ctx.ao_bit > 4: #full precision
            output = input
        else: # multibit
            output = adc(PS=input, minimum=-1, maximum=1, bit=ctx.ao_bit)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if ctx.ao_bit == 1.5:
            quant_grad = gaussian_function(x=input-0.2,sigma=0.02) + gaussian_function(x=input+0.2,sigma=0.02)
        elif ctx.ao_bit > 4: #full precision
            quant_grad = 1
        else:
            quant_grad = impulse_gradient_approximation(x=input, minimum=-1, maximum=1, sigma=0.05, bit=ctx.ao_bit)
        grad_input = grad_output.clone() * quant_grad
        return grad_input, None
QuantInput  = ActivationOutQuantF.apply

class PartialSumQuantF(Function):
    @staticmethod
    def forward(ctx, input, adc_bit=1):
        ctx.save_for_backward(input)
        ctx.adc_bit = adc_bit
        ctx.CiM_row = 128
        output = input.new(input.size())
        if 4 < ctx.adc_bit: # ADC not applied
            output = input
        elif 1 < ctx.adc_bit <= 4: # 2 bit ADC, 3bit ADC and 4bit ADC
            output = adc(PS=input, minimum=-ctx.CiM_row, maximum=ctx.CiM_row, bit=ctx.adc_bit)
            #output = log_adc(x=input, bits=ctx.adc_bit)
        elif ctx.adc_bit == 1: # Sense Amplifier
            output[input >  0] =  ctx.CiM_row
            output[input <= 0] = -ctx.CiM_row

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if 4 < ctx.adc_bit: # ADC not applied
            quant_grad = 1
        else:
            sigma      = [8,8,32,32] #optimized sigmas at 1bit, 2bit, 3bit and 4bit.
            quant_grad = impulse_gradient_approximation(x=input, minimum=-ctx.CiM_row, maximum=ctx.CiM_row, sigma=sigma[ctx.adc_bit-1], bit=ctx.adc_bit)
            #quant_grad =log_impulse_gradient_approximation(x=input, sigma=sigma, bits=ctx.adc_bit)

        grad_input = grad_output.clone() * quant_grad

        return grad_input, None

QuantSum = PartialSumQuantF.apply
########################################################################################

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function): 
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

######################################## quant_sum layer ########################################

class BinarizeLinear_sp(nn.Linear):
    def __init__(self, in_features, out_features, level, *kargs, **kwargs):
        super(BinarizeLinear_sp, self).__init__(in_features, out_features,*kargs, **kwargs)
        self.CiM_row      = 128
        self.in_features  = in_features
        self.level        = level
        self.in_features  = in_features
        self.out_features = out_features
        
    def forward(self, input):
        input.data = QuantInput(input.data)

        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = QuantWeight(self.weight.org)
        
        if self.in_features == 784:
            out = nn.functional.linear(input, self.weight)
            
        if self.in_features == 512:
            input.data       = QuantInput(input.data)
            self.weight.data = QuantWeight(self.weight.data)
            input_list  = list(torch.split(input,       self.CiM_row, dim=1))
            weight_list = list(torch.split(self.weight, self.CiM_row, dim=1))
            out = 0
            for inputIter, weightIter in zip(input_list, weight_list):
                partial_sum           = nn.functional.linear(inputIter, weightIter)
                #partial_sum_quantized = partial_sum
                partial_sum_quantized = QuantSum(partial_sum)
                out += partial_sum_quantized

        if not self.bias is None:
            self.bias.org  = self.bias.data.clone()
            out           += self.bias.view(1, -1).expand_as(out)

        return out

class Conv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)  

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, ao_bit=1, w_bit=1, adc_bit=1, **kwargs, ):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.ao_bit  = ao_bit
        self.w_bit   = w_bit
        self.adc_bit = adc_bit
    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        
        out = 0
        if input.size(1) != 3:
            input.data           = QuantInput(input.data,        self.ao_bit)
            self.weight.data     = QuantWeight(self.weight.data, self.w_bit)
            CiM_row              = 128
            channelSliceWord     = int(CiM_row/9)
            divided_inputTensor  = list(torch.split(input,       channelSliceWord, dim=1))
            divided_weightTensor = list(torch.split(self.weight, channelSliceWord, dim=1))

            for inputIter, weightIter in zip(divided_inputTensor, divided_weightTensor):
                partial_sum = nn.functional.conv2d(inputIter, weightIter, None, self.stride,
                                                   self.padding, self.dilation, self.groups)
                
                partial_sum_quantized = QuantSum(partial_sum, self.adc_bit)
                out += partial_sum_quantized
        else:
            out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out

class BinarizeLinear(nn.Linear): 
    def __init__(self, *kargs, last_layer=False, ao_bit=1, w_bit=1, adc_bit=1,**kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.is_last_layer = last_layer
        self.CiM_row = 128
        self.ao_bit  = ao_bit
        self.w_bit   = w_bit
        self.adc_bit = adc_bit
    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        out = 0
        if not self.is_last_layer:
            input.data       = QuantInput(input.data, self.ao_bit)
            self.weight.data = QuantWeight(self.weight.data, self.w_bit)
            input_list  = torch.split(input,       self.CiM_row, dim=1)
            weight_list = torch.split(self.weight, self.CiM_row, dim=1)
            for inputIter, weightIter in zip(input_list, weight_list):
                partial_sum           = nn.functional.linear(inputIter, weightIter)
                partial_sum_quantized = QuantSum(partial_sum, self.adc_bit)
                out += partial_sum_quantized
        else:
            out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out
            
 
########################################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_values = torch.linspace(-128, 128, 1000)
    bit_functions = {}
    bits_range = range(1, 5)
    
    for bits in bits_range:
        #adc_values = adc(x_values, minimum=-1, maximum=1 , bit=bits)
    
        sigma = [2,6,5,2.5] 
        impulse_gradient_values = impulse_gradient_approximation(x_values,minimum=-128,maximum=128,sigma=sigma[bits-1],bit=bits)
        
        bit_functions[bits] = {'impulse_gradient_values': impulse_gradient_values}
    
    plt.figure(figsize=(12, 8))
    for bits in bits_range:
        plt.plot(x_values.numpy(), bit_functions[bits]['impulse_gradient_values'].numpy(), label=f'')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    plt.show()
    
