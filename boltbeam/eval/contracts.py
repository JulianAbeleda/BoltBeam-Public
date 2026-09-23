from boltbeam.vocab import Guardrail

# The four guardrails that MUST resolve to PASS before a candidate can promote (see eval/evaluator.py
# _REQUIRED_GUARDRAILS, the enforcement site). memory_fit/fallback/determinism are conditional: they only
# block promotion on a firm FAIL, not on absence, so they are not part of this required set.
def promotion_requirements() -> list[str]:
  return [Guardrail.CORRECTNESS.value, Guardrail.ROUTE_BOUND.value, Guardrail.SPEED.value, Guardrail.ROLLBACK.value]
