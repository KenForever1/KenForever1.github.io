---
title: 我的项目
author: KenForever1
date: 2024-12-15
updated: 2024-12-15
---

## 我的开源项目

<div class="grid cards" markdown>

- __[vmtouch-rs](https://github.com/KenForever1/vmtouch-rs)__

    该项目采用rust语言实现了fincore工具和vmtouch工具。

    fincore工具采用mincore获取文件加载到cache中的pages。fincore只能对单文件以及*文件进行统计，不能递归统计目录。相比之下vmtouch更加强大，可以统计目录中文件加载到内存中的pages，也可以对内存中page进行锁住(lock)、回收(evict)等。

- __[part-downloader](https://github.com/KenForever1/part-downloader-rs)__
  
    采用rust实现的一个命令行工具。将大文件分成多个part下载，然后再合并文件。实现原理，采用了HTTP请求中的RANGE Header指定范围。

- __[nvim-rotate-chars](https://github.com/KenForever1/nvim-rotate-chars)__

    这是一个采用nvim-oxi开发的nvim插件，用于旋转选中的文本字符。

- __[nvim_unicode_converter](https://github.com/KenForever1/nvim_unicode_converter)__

    这是一个采用nvim-oxi开发的nvim插件，用于将unicode转换为中文汉字等字符。

- __[qps_client](https://github.com/KenForever1/qps_client)__

    服务请求 benchmark 工具，用于测试服务的qps和delay。

- __[clang-format-cfg-generator-rs](https://github.com/KenForever1/clang-format-cfg-generator-rs
)__

    clang-format配置文件生成工具，可以生成clang-format的配置文件，也可以生成格式化文件的模版。

- __[F16cpp](https://github.com/KenForever1/F16cpp
)__

    F16.hpp实现了IEEE754-2008标准的Float16类型表示, 以及实现了Float32和Float16类型的互相转换。F16类型的直接比较大小方法。

- __[docker-rs](https://github.com/KenForever1/docker-rs
)__

    使用rust实现简单的docker，熟悉linux cgroups、pivotRoot、overlayfs等。


</div>

<script src="https://giscus.app/client.js"
	data-repo="KenForever1/KenForever1.github.io"
	data-repo-id="R_kgDOGbt1Ww"
	data-category="Announcements"
	data-category-id="DIC_kwDOGbt1W84CahvG"
	data-mapping="pathname"
	data-strict="0"
	data-reactions-enabled="1"
	data-emit-metadata="0"
	data-input-position="bottom"
	data-theme="preferred_color_scheme"
	data-lang="zh-CN"
	crossorigin="anonymous"
	async>
</script>
