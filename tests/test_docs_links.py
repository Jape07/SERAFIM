"""Every relative link of the README and the hand-written docs points to a file that exists
(task A10): the validation cards, the demo script and the model card must never be broken."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ["README.md", "docs/demo.md", "docs/model_card.md", "training/RESULTS.md"]
LINK = re.compile(r"\]\(([^)\s]+)\)")


def relative_links(path: Path) -> list[str]:
    links = LINK.findall(path.read_text(encoding="utf-8"))
    return [link.split("#")[0] for link in links
            if not link.startswith(("http://", "https://", "mailto:", "#"))]  # fmt: skip


@pytest.mark.parametrize("document", DOCUMENTS)
def test_every_relative_link_exists(document):
    path = ROOT / document
    broken = [link for link in relative_links(path) if not (path.parent / link).exists()]
    assert broken == [], f"{document}: {broken}"


def test_the_validation_cards_read_in_two_minutes():
    """The check of task A10: the five targets are explained in about 2 minutes (~140 words a
    minute aloud)."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## How each target was validated")[1].split("\n## ")[0]
    words = re.findall(r"[\w'.±″µ-]+", re.sub(r"\([^)]*\.md\)", "", section))
    assert len(words) <= 300
