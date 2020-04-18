# ncnn

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/nullptr-leo/ncnn-lite/master/LICENSE)

ncnn-lite is a derivation of tencent [ncnn](https://github.com/Tencent/ncnn). It is developed for embedded system without C++ and STL support.

ncnn-lite 是基于腾讯 [ncnn](https://github.com/Tencent/ncnn) 开发的衍生项目，去除了对于 C++ 以及 STL 的信赖，便于在嵌入式平台部署。

---

### Differences from ncnn

* NO C++ and STL dependencies, implemented in pure C
* Only contains core inference library and benchmark tool (NO converters and models)
* Only support Linux, ARM and MIPS platforms (NO Windows support and x86 optimization)
* Only support CPU device (NO Vulkan)

### 与 ncnn 的差异

* 纯 C 实现，不信赖于 C++ 及 STL 库
* 仅包含核心推理库和性能测试工具（不包含模型文件及模型转换等工具）
* 仅支持 Linux, ARM 及 MIPS 平台（不支持 Windows 及 x86 平台的算子优化）
* 仅支持 CPU 推理（不支持 GPU Vulkan）

---

### License

BSD 3 Clause

