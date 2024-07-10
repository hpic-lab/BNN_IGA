# BNN-IGA
BNN-IGA is CiM's ADC loss-aware neural network, which is developed for SA-CiM.\
This neural network algorithm is developed by [HPIC Lab](https://hpic-lab.github.io/).

## Quick Start
You can train/inference Res20 with cmd below.\
### -train
```bash
python main_binary.py --results_dir ./results/resnet20 --save test_model --model resnet_binary
```
### -inference
```bash
python main_binary.py --evaluate ./results/resnet20/test_model/model_best.pth.tar --model resnet_binary
```
More args are defined at main_binary.py\
![image](https://github.com/hpic-lab/BNN_IGA/assets/174296776/45e9c7af-6987-469d-8d12-4d3d9ccae864)\
As you can see above code, actiavtion out, weight, and partial sum quantization bits can be set with arg.
For example, train/inference code with 1-bit actiavtion out, 1-bit weight, and 1-bit partial sum cmd can be defined as:
### -train
```bash
python main_binary.py --results_dir ./results/resnet20 --save test_model --model resnet_binary --ao_bit 1 --w_bit 1 --adc_bit 1
```
### -inference
```bash
python main_binary.py --evaluate ./results/resnet20/test_model/model_best.pth.tar --model resnet_binary  --ao_bit 1 --w_bit 1 --adc_bit 1
```

This code only supports quantization up to 4 bits, and if the bit is set to 5 or higher, it operates in full precision. Additionally, if the bit is set to 1.5, it operates in ternary mode.

## Experiment

### Table 1: Accuracy at 1-bit $ao_q$, 1-bit $w_q$, and k-bits ADC on Various Models
We experimented with representative neural networks, such as MLP-MNIST, VGG9-CIFAR10, Res18-CIFAR10, and Res20-CIFAR10, which targeted $128 \times 128$ CiM array.\
The exact result values of inference accuracies about these models are described in TABLE 1.\
We set the $\sigma$ value of the Gaussian function at $k$-bit ADC quantization function's backward phase as 1-bit: 6, 2-bit: 5, 3-bit: 4, 4-bit: 2.5. 
| **Model** | **k-bit ADC** | **MLP-MNIST** | **VGG9-CIFAR10** | **Res18-CIFAR10** | **Res20-CIFAR10** |
|-----------|---------------|---------------|------------------|------------------|------------------|
| **FP**    | 32-bit        | 98.56%        | 92.50%           | 89.05%           | 89.53%           |
| **BNN**   | 32-bit        | **97.74%**    | **91.58%**       | **88.76%**       | **87.73%**       |
|           | 1-bit         | 63.64%        | 12.37%           | 21.49%           | 14.09%           |
|           | 2-bit         | 96.64%        | 61.90%           | 25.98%           | 20.74%           |
|           | 3-bit         | 96.86%        | 80.08%           | 55.61%           | 40.23%           |
|           | 4-bit         | 97.59%        | 88.14%           | 85.08%           | 80.39%           |
| **IGA**   | 1-bit         | **96.57%**    | **89.94%**       | **86.28%**       | **86.46%**       |
|           | 2-bit         | 96.21%        | 90.08%           | 86.34%           | 86.26%           |
|           | 3-bit         | 97.47%        | 89.61%           | 85.36%           | 85.32%           |
|           | 4-bit         | 97.78%        | 90.33%           | 87.16%           | 87.27%           |


### Table 2: Accuracy at k-bit $ao_q$, k-bit $w_q$, and 1-bit ADC on Res20
Table 2 demonstrates a noticeable increase in accuracy with higher resolutions of $ao$ and $w$, particularly for high resolution of $ao$.\
In this experiment, a 4-bit $ao$, 1-bit $w$, and a 1-bit ADC achieved the highest accuracy of $89.59\%$.\
This accuracy result is almost the same as the full precision of Res20 accuracy($89.53\%$).
| $k$-bit $ao_{q}$ | 1-bit $w_{q}$ | 2-bit $w_{q}$ | 3-bit $w_{q}$ | 4-bit $w_{q}$ |
|------------------|---------------|---------------|---------------|---------------|
| 1-bit            | 86.46%        | 88.09%        | 88.57%        | **89.59%**    |
| 2-bit            | 86.46%        | 87.51%        | 88.05%        | 89.20%        |
| 3-bit            | 86.79%        | 88.01%        | 87.64%        | 88.15%        |
| 4-bit            | 86.35%        | 88.98%        | 88.46%        | 88.81%        |

