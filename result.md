# 实验结果总结

## 概述
本文对比了本文提出的方法（下称 `ours`）与 DnCNN、U-Net 及若干消融模型在图像去噪任务上的性能。从数值指标（PSNR / SSIM）、训练曲线、定性结果和效率（参数量与推理时间）四个方面进行汇总。

## 关键结论
- 在低噪声（σ=15）下，各方法表现接近，`ours` 在细节保留上略优。
- 在中高噪声（σ=25, 50）下，`ours` 的 PSNR/SSIM 优于 DnCNN，尤其在 σ=50 时提升更明显，表明对强噪声的鲁棒性更好。
- 注意力模块（例如 CBAM）能进一步提升性能（PSNR / SSIM），但会带来推理时间的增加。`ours` 在性能、参数量和速度之间取得较好平衡。

## 图注与说明
- [figures/training_curves.png](figures/training_curves.png)：训练/验证损失曲线。`ours` 收敛快速且验证损失更低、更稳定，DnCNN 收敛慢且验证误差较高。
- [figures/ssim_comparison.png](figures/ssim_comparison.png)：SSIM 随噪声等级的对比，`ours` 在 σ=15/25/50 均为最高，σ=50 改进显著，代表结构信息保留更好。
- [figures/results_table.png](figures/results_table.png)：总体数值汇总表（PSNR / SSIM / 参数量 / 推理时间），显示 `ours` 在中高噪声下具有竞争力的指标且参数量小（约 0.60M）。
- [figures/psnr_comparison.png](figures/psnr_comparison.png)：PSNR 对比，`ours` 在 σ=25/50 上优于 DnCNN。
- [figures/denoising_examples_sigma50.png](figures/denoising_examples_sigma50.png)：σ=50 的定性去噪示例，`ours` 能去除大量噪声并恢复整体结构，细节有一定平滑但视觉效果接近干净图。
- [figures/denoising_examples_sigma25.png](figures/denoising_examples_sigma25.png)：σ=25 的示例，`ours` 在细节与结构上表现良好。
- [figures/denoising_examples_sigma15.png](figures/denoising_examples_sigma15.png)：σ=15 的示例，去噪效果最佳，细节保留很到位。
- [figures/efficiency_comparison.png](figures/efficiency_comparison.png)：效率对比，U-Net 参数最多但推理最快；`ours` 参数小但推理时间中等；带 CBAM 的消融版本推理显著更慢。
- [figures/ablation_study.png](figures/ablation_study.png)、[figures/ablation_table.png](figures/ablation_table.png)：消融结果显示注意力模块（尤其 CBAM）在 PSNR/SSIM 上带来正向提升，参数量基本不变。

## 重要数值（摘录自 `results_table.png` / `ablation_table.png`）
- `ours`: PSNR ≈ 33.91 dB，SSIM ≈ 0.901，参数量 ≈ 0.60M，推理时间 ≈ 16.9 ms。
- `Ablation C (CBAM)`: PSNR ≈ 35.02 dB，SSIM ≈ 0.919，推理时间显著增加（≈43.9 ms）。

## 再现与复现提示
- 运行训练/评估脚本：

```bash
python3 main.py --mode test --config configs/your_config.yaml
```

- 若需生成本文图像，确认 `figures/` 下的图片由 `visualize.py` 或 `generate_charts.py` 产生，并按数据路径放置 `checkpoints/` 与 `data/`。

## 建议的后续工作
- 如果目标是部署到实时系统，可进一步优化推理时间（剪枝/量化/轻量化注意力）；
- 若关注主观视觉质量，可针对高频细节设计专门的损失（perceptual / adversarial）；
- 是否需要我把本文图注翻译为英文并生成一个可直接放入论文的 `figures` 段落？

---

文件作者：实验整理自动生成
生成时间：2026-05-29
