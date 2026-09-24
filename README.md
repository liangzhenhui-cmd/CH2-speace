# Chronos-2 广东价差预测（项目2复现版）

本项目把原项目2整理为与项目1相同的结构，并把数据、特征角色、LoRA适配器和历史参考结果放到项目内部。默认参数用于复现项目2保存的效果。

## 每日生产预测（best）

Git版本保留预测代码、依赖、当前数据快照、特征角色和选定的best LoRA适配器。基础模型`amazon/chronos-2`不放入Git，首次运行时由Hugging Face下载并缓存。

每天先把最新源表保存为`data/24点数据.csv`，然后执行：

```powershell
cd D:\chronos2_spread
.\.venv\Scripts\python.exe feature_engineering.py --update
.\.venv\Scripts\python.exe predict_chronos2.py 2026-09-25 --model best --device cpu
```

将日期替换为目标预测日。有CUDA环境时可以把`cpu`改为`cuda`。输出位于`outputs`目录：

```text
2026-09-25-ch2策略.csv
2026-09-25-ch2策略_simple.csv
```

简明表只包含时间、策略和负价差概率。best适配器位置为：

```text
models/chronos2_spread/lora-recent180-w3-lr12e6-r16-a16-b136-s450/lora-adapter
```

## 固定实验设置

- 目标：`spread = 日前价格 - 实时价格`
- 决策时点：D-1 12:00
- 历史截止：D-3 23:00
- 上下文：672小时
- 预测长度：72小时，只评价最后24小时（D日）
- 特征：59项known-future、8项past-only
- 微调：LoRA r=8、alpha=16、dropout=0，30步，seed=42，learning rate=1e-5
- batch：68条变量序列，约等于每步1个日任务
- 训练目标日截止：2026-08-14
- 复现回测：2026-08-15至2026-09-12，共696小时
- 基础模型快照：`amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`

## 目录

- `data/24点数据.csv`：每日同步到项目内的电力源数据。
- `data/weather_21cities_history.csv`：每日同步并补充预报的21城天气源数据。
- `data/chronos2_features.csv`：两张每日源表合成的严格D-3总表。
- `data/feature_roles.json`：known-future/past-only角色清单。
- `chronos2_utils.py`：包含每日更新函数`update_forecast`及预测窗口工具。
- `feature_engineering.py`：更新源表后直接重建严格D-3总表，不再依赖静态中间表。
- `train_chronos2.py`：按项目2参数进行30步LoRA微调。
- `predict_chronos2.py`：单日零样本或LoRA预测。
- `evaluate_chronos2.py`：批量复现零样本、LoRA与D-3基线回测。
- `compare_decision_rules.py`：比较q50、负价差概率阈值和分位数积分期望三种1.2/0.8规则。
- `verify_reproduction.py`：把新输出与原项目2结果逐列比较。
- `models/chronos2_spread/lora-adapter`：原项目2训练好的适配器。
- `outputs/reference_project2`：原项目2保存的参考结果，只用于验收。

## 环境与运行

已配置项目独立虚拟环境 `.venv`（Python 3.11）。在 PowerShell 中运行：

```powershell
cd D:\chronos2_spread
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

每天先手工把最新24点表复制为：

```text
D:\chronos2_spread\data\24点数据.csv
```

然后更新21城天气并重建总表：

```powershell
python feature_engineering.py --update
```

默认不会复制或覆盖24点表。天气从
`D:\heyuan_predict\weather\weather\weather_21cities_history.csv`同步。

只有需要从其他位置自动复制24点表时，才显式指定：

```powershell
python feature_engineering.py --update --power-source "D:\其他目录\24点数据.csv"
```

天气缺少到明天的数据时，`update_forecast`会从Open-Meteo补齐。只同步现有文件、不访问天气API：

```powershell
python feature_engineering.py --update --no-weather-api
```

如果两张源表已经放在本项目`data`目录，只重建总表：

```powershell
python feature_engineering.py
```

新总表会延伸到电力预测和21城完整天气共同覆盖的最后一个小时。未来目标和实际值允许为空；59项未来已知特征必须完整。训练只使用目标与历史实际值完整的日期，预测时未来区间只读取59项known-future特征。

复现微调：

```powershell
python train_chronos2.py
```

单日零样本与LoRA预测：

```powershell
python predict_chronos2.py 2026-09-12 --model zero-shot
python predict_chronos2.py 2026-09-12 --model lora
```

正式预测按q50正负选择1.2或0.8，同时输出`negative_probability`和
`direction_confidence`用于解释。概率不参与策略选择。

复现原回测并与原结果比较：

```powershell
python evaluate_chronos2.py --start 2026-08-15 --end 2026-09-12 --models both
python verify_reproduction.py
```

比较三种二选一策略（每种只输出1.2或0.8）：

```powershell
python compare_decision_rules.py --model lora --probability-threshold 0.55
```

逐小时文件同时输出负价差概率和方向置信度；q50规则只展示该概率，不用概率改变决策。概率阈值应在独立验证期选择，不能根据最终测试期最高收益反向挑选。

原结果为：零样本MAE 72.6841、方向准确率58.3333%、理论收益3178.4；LoRA 30步MAE 72.3416、方向准确率58.6207%、理论收益3220.0。

复现时默认使用CPU和8线程，与项目2一致。改用CUDA可能产生很小的数值差异。基础模型没有复制到项目目录，首次运行需要本机已有对应Hugging Face缓存或联网下载；适配器已经复制到本项目。

若 PowerShell 禁止激活脚本，可直接使用 `.\.venv\Scripts\python.exe predict_chronos2.py 2026-09-12 --model lora`。IDE 解释器请选择 `D:\chronos2_spread\.venv\Scripts\python.exe`。

