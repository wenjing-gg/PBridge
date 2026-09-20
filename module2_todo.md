# Lingshu-PBridge Image Preprocess Validation

## 目标

验证当前 Lingshu-7B backbone 的图像输入是否保留空间视觉 token，并排除：

1. 原始数据过小；
2. 数据加载阶段提前缩图；
3. Processor 像素预算异常；
4. 所有图片被压成同一低分辨率网格；
5. Transformers 版本兼容导致异常 resize。

## 验证项

- 记录 Processor 类、Transformers版本、`min_pixels`、`max_pixels`、
  `patch_size`、`merge_size` 和 `size`。
- 随机抽取至少20张图片，对比原始尺寸、加载后PIL尺寸和Processor输入尺寸。
- 对完整413题统计 `image_grid_thw`、`pixel_values.shape` 和最终视觉token数。
- 检查 checkpoint 中的 `preprocessor_config.json`、`config.json`、
  `processor_config.json` 和 `tokenizer_config.json`。
- 选择一张大图，分别测试默认预算、checkpoint显式预算、TODO建议预算和
  强制单token预算。
- 判断当前输入是否适合后续空间注意力分析。

## 输出

结果写入：

```text
module2_probe/outputs/image_preprocess_check/
├── processor_config.txt
├── image_size_statistics.json
├── image_grid_statistics.json
├── pixel_budget_comparison.json
├── sample_debug.jsonl
└── conclusion.md
```

最终结论必须回答：

1. 原始图片尺寸范围；
2. 数据加载是否提前缩图；
3. 当前像素预算；
4. 完整测试集视觉网格和token数分布；
5. 是否存在单视觉token缺陷；
6. 是否需要重新配置分辨率并重新训练PBridge。
