"""Grade answer content with explicit, per-question alternatives.

Raw text and strict scores remain visible. No fuzzy matching, global numeric
string conversion, or output-format compliance is implied by a content pass.
"""
from __future__ import annotations
import json


def same_answer(actual, expected) -> bool:
  # JSON booleans must not compare equal to Python's integer 0/1.
  if isinstance(actual, bool) or isinstance(expected, bool):
    return type(actual) is type(expected) and actual == expected
  if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
    return actual == expected
  if type(actual) is not type(expected):
    return False
  if isinstance(expected, list):
    return len(actual) == len(expected) and all(same_answer(a, b) for a, b in zip(actual, expected))
  if isinstance(expected, dict):
    return actual.keys() == expected.keys() and all(same_answer(actual[k], expected[k]) for k in expected)
  return actual == expected


def grade_response(case: dict, text: str, token_ids=None, *, thinking=False, finish_reason=None) -> dict:
  actual = None
  error = None
  final_text = text
  if thinking:
    _, separator, final_text = text.rpartition('</think>')
    if not separator:
      error = 'thinking did not finish: missing </think>'
  if finish_reason == 'length':
    error = 'generation reached token limit'
  try:
    if error:
      raise ValueError(error)
    # Preserve the existing content extraction (including fenced JSON).
    answer = json.JSONDecoder().raw_decode(final_text[final_text.index('{'):])[0]
    if not isinstance(answer, dict) or 'answer' not in answer:
      raise ValueError('missing answer key')
    actual = answer['answer']
  except (ValueError, TypeError) as exc:
    error = str(exc)
  strict = error is None and same_answer(actual, case['answer'])
  alternatives = case.get('accepted_answers', [])
  accepted = error is None and any(same_answer(actual, value) for value in alternatives)
  result = dict(prompt=case['prompt'], expected=case['answer'], actual=actual,
                strict_passed=strict, passed=strict or accepted,
                match='exact' if strict else 'accepted_answer' if accepted else 'failed',
                accepted_answers=alternatives, text=text, final_text=final_text,
                thinking=thinking, finish_reason=finish_reason)
  if error is not None:
    result['parse_error'] = error
  if token_ids is not None:
    result['token_ids'] = token_ids
  return result
