# 积雪集合预报的加权 CRPS 归约与成员归因

积雪集合成员和校准观测的轮次交换资料。 仓库记录均为本地合成资料，不依赖外部数据服务。

## 数据约定

entries 逐列是到达序号、中心、稳定成员、修订、格点、雪水等效和质量标记。主键按轮次、中心、成员和格点解释，修订在该键内递增。质量 false 取代旧资料并剔除，时间超过 seal 的修订不属于正式结果。

中心权重在各格点参与的中心之间重新归一，每中心再在有效成员间均分。成对项按有序不同成员之权重乘积和归一，CRPS 为观测绝对偏差项减去成对结果的一半。聚合依 grid_area 在可交付且有观测的格点上加权。

query 是校准系统的收件时点。read_windows 是预览后修订可访问的原始输入组。未列出的早期新接收资料已完成入库，由计算侧保有此前收到的内容或可归约资料。正式运行以其他中心的原始成员读取数 0、同一更新最多 128 次其他中心摘要页读为准，每页最多 64 个数值及其关联标识。准备输入时和逐段读取梯度时的预算与增量更新分别计算。g3 的观测有效位为 false，不参加结算，而 g1 的成员值与观测值同为 0.4 是有效的相等值。

## 读取与核对

在 Python 3.11 环境执行 `python -m pip install -r requirements.txt` 安装数值依赖。
`python -m unittest discover -s tests -v` 读取 `data/exchange.json` 并核对资料结构与关键输入关系。

## 包结构

```
snowcrps/
  crps.py      带权经验 CRPS 与解析导数（O(n log n)，不构造成员对矩阵）
  exchange.py  交换资料解析、按轮次取修订、按窗口读取与预算记账
  core.py      按格点结算、增量更新、梯度交接、封存回查
```

### 结算

```python
import json
from pathlib import Path
from snowcrps import load_exchange, Ledger

exchange = load_exchange(json.loads(Path("data/exchange.json").read_text()))
ledger = Ledger(exchange)
book = ledger.prepare()               # 默认按 seal 轮次结算全部格点
book["g1"].loss                       # 加权 CRPS
book["g1"].members                    # 归因：中心、成员、权重、名次、值组
book["g1"].centers                    # 中心权重归属（已按参与中心归一）
book.aggregate(exchange.grid_area)    # 只在可交付且有观测的格点上按面积加权
```

每个格点结算的 `status` 为 `delivered` / `insufficient_support` /
`no_observation` 之一。成对项需要至少两个正权重成员；只有一个有效成员时
不给出损失。`availability` 区分 `pending`（截止早于 seal，后续可补齐）与
`formal`（已到 seal，按承诺成员集合正式缺失）。

### 增量更新与读取预算

```python
from snowcrps import RevisionReader

reader = RevisionReader(exchange, own_center="north")
report = ledger.apply_change("g1", "north", cutoff=6, arrival=5, reader=reader)
report.membership_changed, report.ranking_changed, report.weights_changed
report.own_raw_members, report.other_center_raw_members
```

* 只有 `read_windows` 列出的到达才允许重读该中心的原始成员；读取严格限定在
  该中心受影响的格点，其他中心原始成员读取恒为 0。
* 读取量等于该到达涉及的成员数，不随其他中心成员总数增长。
* `verify_other_centers=True` 时用摘要页核对其他中心，页数受
  `max_update_work.other_summary_page_reads` 约束。

### 校准器交接

```python
batch = ledger.build_batch(book, grids=["g1"])
packet = batch.packet("g1")           # 绑定 token 与 epoch
ledger.accept_packet(packet)          # 旧状态的梯度包会抛 StaleGradientError
ledger.fetch_gradients("g1", 6, members=[("north", "n1")], reader=reader)
ledger.fetch_gradient_segments(["g1"], 6, reader=reader)
```

`Settlement.exposed()` 按口径返回可交给校准器的量：正式口径含梯度与分解项，
预览口径只含损失、成员归因与状态。聚合分母不含失效格点。

### 封存与回查

```python
sealed = ledger.seal(6)               # 记录承诺成员集合与结算快照
ledger.recall_sealed("g1", verify=True)
```

`recall_sealed` 用承诺成员集合重建结算并与快照比较；不一致直接报错。
