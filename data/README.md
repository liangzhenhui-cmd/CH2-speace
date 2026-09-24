# 每日数据目录

此目录包含每日预测需要的业务数据：

- `24点数据.csv`：更新后的电力源表，预测目标日 D 至少需要包含到 D-3 的完整实际值，并包含到 D 的未来已知字段。
- `weather_21cities_history.csv`：21城天气历史种子文件；`feature_engineering.py --update` 会联网补齐天气预报。

`chronos2_features.csv` 和 `feature_roles.json` 由特征脚本生成。仓库保留当前快照，克隆后可以直接预测；每日更新后可按需要提交新的数据快照。

更新并预测：

```powershell
.\.venv\Scripts\python.exe feature_engineering.py --update
.\.venv\Scripts\python.exe predict_chronos2.py YYYY-MM-DD --model best --device cpu
```

如果天气历史源不在默认位置，更新时增加：

```powershell
--weather-source "D:\你的目录\weather_21cities_history.csv"
```
