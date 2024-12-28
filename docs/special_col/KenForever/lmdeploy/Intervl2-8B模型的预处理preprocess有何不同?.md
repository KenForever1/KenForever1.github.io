
Intervl2-8B模型的预处理preprocess有何不同?

Intervl2-8B模型会对图片进行切图，切成448x448的子图送入模型。和QwenVL不一样，QwenVL可以输入任意大小图片。

在~/lmdeploy/vl/model/builder.py中根据模型配置hf_config，获取model模块，调用预处理build_preprocessor函数。
```python
_, hf_config = get_model_arch(model_path)
kwargs = dict(model_path=model_path,
                with_llm=with_llm,
                max_memory=max_memory,
                hf_config=hf_config,
                backend=backend)
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

```

查看hf_config，模型配置信息。get_model_arch函数中的逻辑：

```python
from transformers import AutoConfig

model_path = "/workspace/lm_deploy_repos/InternVL2-8B"
cfg = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
_cfg = cfg.to_dict()
print(_cfg)
```

可以看到，部分打印内容如下：
```bash
'use_backbone_lora': 0, 'use_llm_lora': 0, 'select_layer': -1, 'force_image_size': 448, 'downsample_ratio': 0.5, 'template': 'internlm2-chat', 'dynamic_image_size': True, 'use_thumbnail': True, 'ps_version': 'v2', 'min_dynamic_patch': 1, 'max_dynamic_patch': 12}
```

在*~/lmdeploy/vl/model/internvl.py*中，如果'dynamic_image_size': True，就会走v2的前处理逻辑：

```python
if dynamic_image_size or image_processor is None:
    logger.info('using InternVL-Chat-V1-5 vision preprocess')
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    input_size = self.config.vision_config.image_size
    self.transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')
                    if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    self.processor = self._preprocess_v1_5
    self._forward_func = self._forward_v1_5
```

```python
def _preprocess_v1_5(self, image, params=None):
    image_res = {'low': 6, 'medium': 12, 'high': 24}
    max_num = params.get('max_dynamic_patch')
    if max_num is None or not isinstance(max_num, int):
        res_key = params.get('detail', 'default')
        max_num = image_res.get(res_key, self.config.max_dynamic_patch)
    out = dynamic_preprocess(
        image,
        min_num=self.config.min_dynamic_patch,
        max_num=max_num,
        image_size=self.config.vision_config.image_size,
        use_thumbnail=self.config.use_thumbnail)
    pixel_values = [self.transform(x) for x in out]
    # (patch) x c x h x w
    pixel_values = torch.stack(pixel_values)
    return pixel_values
```

dynamic_preprocess函数在[InternVL2-8B/summary](https://www.modelscope.cn/models/OpenGVLab/InternVL2-8B/summary)这里Inference with Transformers小节也有介绍。

和QwenVL不一样，QwenVL可以输入任意大小图片。

多模态大模型处理图像时**将图像切分成多个448*448的子图**，**每个448*448的子图是x个token，子图切分规则：**

图像宽为 `width`、图像高为 `height`、图像宽切分块数 `m`、图像高切分块数`n`、最大切分块数 `block_count_max`、最小切分块数 `block_count_min`，切分的 `m`、`n` 同时满足以下条件:

1. `block_count_min <= m * n <= block_count_max`
    
2. `m / n`与 `width / height`的差值最小

token计算公式：`token = m * n * x`

lmdeploy中实现在[model/internvl.py](https://github1s.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/model/internvl.py)。