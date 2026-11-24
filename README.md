# pairwise-towerdefense-repath-stale

## 领域

游戏、图形与多媒体

## 主题

塔防动态重寻路与过期结果

## 说明

在途任务作废、网格版本校验与占用格确定性分配。

初始状态下 `gameplay/towerdefense_repath_stale.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
