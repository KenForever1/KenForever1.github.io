
在lmdeploy中支持了很多模型的推理，这些模型是如何注册给lmdeploy框架的呢？
lmdeploy框架又是如何调用到正确的模型推理的呢?

> MMEngine 实现的注册器可以看作一个映射表和模块构建方法（build function）的组合。映射表维护了一个字符串到类或者函数的映射，使得用户可以借助字符串查找到相应的类或函数，例如维护字符串 "ResNet" 到 ResNet 类或函数的映射，使得用户可以通过 "ResNet" 找到 ResNet 类；而模块构建方法则定义了如何根据字符串查找到对应的类或函数以及如何实例化这个类或者调用这个函数，例如，通过字符串 "bn" 找到 nn.BatchNorm2d 并实例化 BatchNorm2d 模块；又或者通过字符串 "build_batchnorm2d" 找到 build_batchnorm2d 函数并返回该函数的调用结果。

### lmdepoy中调用模块
使用案例：

```bash
https://github1s.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/model/builder.py#L65
```

如果VISION_MODELS中的module和hf_config匹配，就传递参数调用模块。调用模块的相关函数。
```python
from lmdeploy.vl.model.base import VISION_MODELS
def load_vl_model(model_path: str,
                  backend: str,
                  with_llm: bool = False,
                  backend_config: Optional[Union[TurbomindEngineConfig,
                                                 PytorchEngineConfig]] = None):
    ...
    for name, module in VISION_MODELS.module_dict.items():
    try:
        if module.match(hf_config):
            logger.info(f'matching vision model: {name}')
            model = module(**kwargs)
            model.build_preprocessor()
            # build the vision part of a VLM model when backend is
            # turbomind, or load the whole VLM model when `with_llm==True`
            if backend == 'turbomind' or with_llm:
                model.build_model()
            return model
    except Exception as e:
        logger.error(f'build vision model {name} failed, {e}')
        raise

    raise ValueError(f'unsupported vl model with config {hf_config}')
                                        

```

### 如何注册model的呢？

在**vl/models**目录中可以看到注册的很多模型，比如qwen、internvl等。

在**lmdeploy/vl/model/internvl.py**文件中，可以看到定义InternVLVisionModel类时，通过@VISION_MODELS.register_module()进行了注册。

```python
@VISION_MODELS.register_module()
class InternVLVisionModel(VisonModel):
    """InternVL vision model."""

    _arch = 'InternVLChatModel'

    def __init__(self,...):
        ....
```

VISION_MODELS就是mmengine中Registry的一个实例类型。

```python
from mmengine import Registry
VISION_MODELS = Registry('vision_model')
```


在你的项目中如果你也需要根据配置，调用不同的模块，也可以采用这种方法。注册绑定，最简单的就是建立字符串和类的映射关系。

在这里lmdeploy使用了[mmengine中的Registry模块](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html
)。支持的功能更加全面，包括：

+ 模块的注册和调用
+ 函数的注册和调用
+ 模块间父子关系建立，如果子节点找不到，就去父节点中找对应模块调用
+ 兄弟节点关系建立

简单使用例子：
使用注册器管理代码库中的模块，需要以下三个步骤。

+ 创建注册器

+ 创建一个用于实例化类的构建方法（可选，在大多数情况下可以只使用默认方法）

+ 将模块加入注册器中

假设我们要实现一系列激活模块并且希望仅修改配置就能够使用不同的激活模块而无需修改代码。
```python
from mmengine import Registry
# scope 表示注册器的作用域，如果不设置，默认为包名，例如在 mmdetection 中，它的 scope 为 mmdet
# locations 表示注册在此注册器的模块所存放的位置，注册器会根据预先定义的位置在构建模块时自动 import
ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])


import torch.nn as nn

# 使用注册器管理模块
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x

@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call ReLU.forward')
        return x

@ACTIVATION.register_module()
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Softmax.forward')
        return x

print(ACTIVATION.module_dict)
# {
#     'Sigmoid': __main__.Sigmoid,
#     'ReLU': __main__.ReLU,
#     'Softmax': __main__.Softmax
# }
```

注册后，使用：
```python
import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call Sigmoid.forward
print(output)


如果我们想使用 ReLU，仅需修改配置。

act_cfg = dict(type='ReLU', inplace=True)
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call ReLU.forward
print(output)

```
进阶功能参考：https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html。
