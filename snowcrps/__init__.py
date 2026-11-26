"""按格点组的加权 CRPS 结算与成员归因。

公开入口：

* :func:`snowcrps.load_exchange` —— 读取并校验交换资料；
* :class:`snowcrps.RevisionReader` —— 按中心与格点范围读取，分口径记账；
* :class:`snowcrps.Ledger` —— 结算、增量更新、封存回查与梯度交接；
* :func:`snowcrps.weighted_crps` —— 与账本无关的加权 CRPS 核。
"""

from .core import (
    DELIVERED,
    EXPOSURE_FORMAL,
    EXPOSURE_FIELDS,
    EXPOSURE_PREVIEW,
    FORMAL,
    INSUFFICIENT_SUPPORT,
    NO_OBSERVATION,
    PENDING,
    SEALED,
    Aggregate,
    CenterShare,
    CenterSlice,
    GradientPacket,
    Ledger,
    LedgerError,
    MemberShare,
    SealedRound,
    Settlement,
    SettlementBook,
    StaleGradientError,
    TrainingBatch,
    UpdateReport,
    read_delivery_rows,
)
from .crps import CrpsResult, weighted_crps, weighted_crps_naive
from .exchange import (
    TEXT_FIELDS,
    BudgetError,
    Exchange,
    ExchangeValidationError,
    ReadWindow,
    Revision,
    RevisionReader,
    load_exchange,
    summarize_revision,
)

__all__ = [
    "DELIVERED",
    "EXPOSURE_FIELDS",
    "EXPOSURE_FORMAL",
    "EXPOSURE_PREVIEW",
    "FORMAL",
    "INSUFFICIENT_SUPPORT",
    "NO_OBSERVATION",
    "PENDING",
    "SEALED",
    "Aggregate",
    "BudgetError",
    "CenterShare",
    "CenterSlice",
    "CrpsResult",
    "Exchange",
    "ExchangeValidationError",
    "GradientPacket",
    "Ledger",
    "LedgerError",
    "MemberShare",
    "ReadWindow",
    "Revision",
    "RevisionReader",
    "SealedRound",
    "Settlement",
    "SettlementBook",
    "StaleGradientError",
    "TEXT_FIELDS",
    "TrainingBatch",
    "UpdateReport",
    "load_exchange",
    "read_delivery_rows",
    "summarize_revision",
    "weighted_crps",
    "weighted_crps_naive",
]
