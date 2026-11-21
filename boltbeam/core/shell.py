"""A command as one shell line and back: shell_word quotes a word (in $'...' when it holds a control character), and
split_command, its inverse, reads such a line into words. Every reader of a server command BoltBeam wrote or a user
pasted uses split_command: shlex alone reads no $'...' (a leading $ and a literal backslash-n come back)."""
from __future__ import annotations

import re
import shlex
import unicodedata
from typing import Any

_CTRL = {"\\": "\\\\", "'": "\\'", "\n": "\\n", "\t": "\\t", "\r": "\\r"}
_UNESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "?": "?", "a": "\a", "b": "\b", "e": "\x1b",
          "E": "\x1b", "f": "\f", "v": "\v"}
_ESC = re.compile(r"\\(x[0-9A-Fa-f]{1,2}|u[0-9A-Fa-f]{1,4}|U[0-9A-Fa-f]{1,8}|[0-7]{1,3}|.)", re.S)


def shell_word(a:Any) -> str:
  """One word of a command as a shell reads it back: as is when it is plain, single-quoted otherwise, and in $'...'
  (bash, zsh) when it holds a control character (a message ending in a newline), so the command stays one line."""
  s = str(a)
  if not any(unicodedata.category(c) == "Cc" for c in s):
    return shlex.quote(s)
  return "$'" + "".join(_CTRL.get(c) or (f"\\u{ord(c):04x}" if unicodedata.category(c) == "Cc" else c) for c in s) + "'"


def command_line(env:dict[str, Any], argv:list[Any]) -> str:
  """The env block and the argv as one shell line, each word quoted where it needs it (shell_word)."""
  return " ".join([*(f"{k}={shell_word(v)}" for k, v in env.items()), *map(shell_word, argv)])


def _ansi_c(body:str) -> str:
  """The text of a $'...' body, as bash decodes it: \\n \\t \\r \\\\ \\' and the rest, \\xHH, \\uHHHH, \\UHHHHHHHH, \\nnn."""
  def one(m:re.Match) -> str:
    e = m.group(1)
    if e[0] in "xuU" and len(e) > 1:
      return chr(int(e[1:], 16))  # past U+10FFFF: ValueError, as shlex raises for a bad line
    if e[0] in "01234567":
      return chr(int(e, 8))
    return _UNESC.get(e, "\\" + e)
  return _ESC.sub(one, body)


def split_command(text:str) -> list[str]:
  """The words of a shell line (shlex's POSIX rules), with $'...' words decoded and a backslash-newline read as a space:
  split_command(command_line(env, argv)) == [K=V ..., *argv]."""
  out, i, n = [], 0, len(text)
  while i < n:
    c = text[i]
    if c == "'":  # a single-quoted run: as is
      j = text.find("'", i + 1)
      j = n - 1 if j < 0 else j
      out.append(text[i:j + 1])
      i = j + 1
    elif c == '"':  # a double-quoted run: to the first unescaped quote
      j = i + 1
      while j < n and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
      out.append(text[i:j + 1])
      i = j + 1
    elif c == "\\" and text[i + 1:i + 2] == "\n":
      out.append(" ")
      i += 2
    elif c == "\\":
      out.append(text[i:i + 2])
      i += 2
    elif c == "$" and text[i + 1:i + 2] == "'":  # ANSI-C: decode, then quote for shlex
      j = i + 2
      while j < n and text[j] != "'":
        j += 2 if text[j] == "\\" else 1
      if j >= n:
        raise ValueError("No closing quotation")  # as shlex says of an unclosed quote
      out.append(shlex.quote(_ansi_c(text[i + 2:j])))
      i = j + 1
    else:
      out.append(c)
      i += 1
  return shlex.split("".join(out))
