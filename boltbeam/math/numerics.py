"""Numeric comparison policies, separate from workload and engine logic.

Kernel correctness permits float32 reduction-order differences. Accounting
compares derived double-precision quantities; it must not inherit that looser
kernel threshold. Display rounding remains a property of the displayed field.
"""
TOLERANCE = 1e-3  # max |gpu - cpu| / max |cpu|
ACCOUNTING_RTOL = 1e-8
ACCOUNTING_ATOL = 1e-6
INTERVAL_ATOL = 1e-9  # milliseconds: kernel interval conservation
