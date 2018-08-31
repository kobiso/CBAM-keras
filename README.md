# CBAM-Keras
This is a Keras implementation of ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/pdf/1807.06521).
This repository includes the implementation of ["Squeeze-and-Excitation Networks"](https://arxiv.org/pdf/1709.01507) as well, so that you can train and compare among base CNN model, base model with CBAM block and base model with SE block.

## CBAM: Convolutional Block Attention Module
**CBAM** proposes an architectural unit called *"Convolutional Block Attention Module" (CBAM)* block to improve representation power by using attention mechanism: focusing on important features and supressing unnecessary ones.
This research can be considered as a descendant and an improvement of ["Squeeze-and-Excitation Networks"](https://arxiv.org/pdf/1709.01507).

### Diagram of a CBAM_block
<div align="center">
  <img src="https://github.com/kobiso/CBAM-keras/blob/master/figures/overview.png">
</div>

### Diagram of each attention sub-module
<div align="center">
  <img src="https://github.com/kobiso/CBAM-keras/blob/master/figures/submodule.png">
</div>

### Classification results on ImageNet-1K

<div align="center">
  <img src="https://github.com/kobiso/CBAM-keras/blob/master/figures/exp4.png">
</div>

<div align="center">
  <img src="https://github.com/kobiso/CBAM-keras/blob/master/figures/exp5.png"  width="750">
</div>

## Prerequisites
- Python 3.x
- Keras

## Prepare Data set
This repository use [*Cifar10*](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
When you run the training script, the dataset will be automatically downloaded.
(Note that you **can not run Inception series model** with Cifar10 dataset, since the smallest input size available in Inception series model is 139 when Cifar10 is 32. So, try to use Inception series model with other dataset.)

## CBAM_block and SE_block Supportive Models
You can train and test base CNN model, base model with CBAM block and base model with SE block.
You can run **CBAM_block** or **SE_block** added models in the below list.

- Inception V3 + CBAM / + SE
- Inception-ResNet-v2 + CBAM / + SE
- ResNet_v1 + CBAM / + SE (ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet164, ResNet1001)
- ResNet_v2 + CBAM / + SE (ResNet20, ResNet56, ResNet110, ResNet164, ResNet1001)
- ResNeXt + CBAM / + SE
- MobileNet + CBAM / + SE
- DenseNet + CBAM / + SE (DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264)

### Change *Reduction ratio*
To change *reduction ratio*, you can set `ratio` on `se_block` and `cbam_block` method in `models/attention_module.py`

## Train a Model
You can simply train a model with `main.py`.

1. Set a model you want to train.
    - e.g. `model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)`  
2. Set attention_module parameter
    - e.g. `attention_module = 'cbam_block'`
3. Set other parameter such as *batch_size*, *epochs*, *data_augmentation* and so on.
4. Run the `main.py` file
    - e.g. `python main.py`

## Related Works
- Blog: [CBAM: Convolutional Block Attention Module](https://kobiso.github.io//research/research-CBAM/)
- Repository: [CBAM-TensorFlow](https://github.com/kobiso/CBAM-tensorflow)
- Repository: [CBAM-TensorFlow-Slim](https://github.com/kobiso/CBAM-tensorflow-slim)
- Repository: [SENet-TensorFlow-Slim](https://github.com/kobiso/SENet-tensorflow-slim)

## Reference
- Paper: [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521)
- Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- Repository: [Keras: Cifar10 ResNet example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
- Repository: [keras-squeeze-excite-network](https://github.com/titu1994/keras-squeeze-excite-network)
  
## Author
Byung Soo Ko / kobiso62@gmail.com
