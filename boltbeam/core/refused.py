"""Refused: a command that cannot answer (an unknown target, a folder that is not a run, a chip with no ceiling).

One exception for every screen command and the limit below them (roofline/limit.py), so a caller catches one class.
"""


class Refused(Exception):
  """A command that cannot answer: an unknown target, a folder that is not a run, a chip with no ceiling."""
