"""A3: frozen A2 + the JEV API, asked typed questions over a JSON evidence state per clip.

Submodules: `client` (the `JevClient` interface + dry-run/typesafe implementations),
`cache` (the committed on-disk request/response cache), `state` (the label-free evidence
builder), `questions` (the versioned question set). See each module's docstring.
"""

from __future__ import annotations
