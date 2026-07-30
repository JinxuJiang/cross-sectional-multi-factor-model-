# LightGBM V2 调参

本目录提供两个脚本：

- `tune_lgbm_v2.py`：在固定代表季度上用 Optuna 搜索参数。
- `validate_lgbm_v2_params.py`：对 top 参数做非重合季度验证和邻近扰动稳定性检查。

默认调参季度：

```text
2020Q2, 2021Q4, 2022Q2, 2022Q4, 2024Q1, 2025Q4
```

输出目录：

```text
03模型训练层/tuning/tuning_results/{study_name}/
```

保留文件：

```text
study_results.csv
top_trials.csv
tuning_summary.md
validation_results.csv
final_recommended_config.yaml
```

缓存目录：

```text
03模型训练层/tuning/.cache/
```

缓存默认在脚本结束后删除；需要保留时加 `--keep-cache`。

## 用法

调参：

```powershell
cd C:\Users\蒋大王\Desktop\量化\截面多因子模型\03模型训练层

python tuning\tune_lgbm_v2.py `
  --base-config configs\horizon20_config.yaml `
  --study-name lgbm20_v2_001 `
  --n-trials 20
```

验证：

```powershell
python tuning\validate_lgbm_v2_params.py `
  --study-name lgbm20_v2_001 `
  --top-k 3
```

最终用于完整训练的配置：

```text
03模型训练层/tuning/tuning_results/lgbm20_v2_001/final_recommended_config.yaml
```
