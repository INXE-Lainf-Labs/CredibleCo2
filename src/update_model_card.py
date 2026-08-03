#!/usr/bin/env python3
"""Synchronize the root model card from the canonical second-revision copy.

The former implementation regenerated a pre-revision model card from historical
result files and could reintroduce outdated metrics and causal wording. The
canonical source is now ``submission/r2/MODEL_CARD.md``.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission" / "r2" / "MODEL_CARD.md"
TARGET = ROOT / "MODEL_CARD.md"


def main() -> None:
    content = SOURCE.read_text(encoding="utf-8")
    TARGET.write_text(content, encoding="utf-8")
    print(f"Synchronized {TARGET.relative_to(ROOT)} from {SOURCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
