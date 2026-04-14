from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


def parse_markdown(markdown_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    in_code = False
    code_lines: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip("\n")

        if line.strip().startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            blocks.append(("blank", ""))
            continue

        if stripped.startswith("# "):
            blocks.append(("h1", stripped[2:].strip()))
        elif stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
        elif stripped.startswith("- "):
            blocks.append(("bullet", stripped[2:].strip()))
        elif stripped[0].isdigit() and ". " in stripped:
            num_end = stripped.find(". ")
            if num_end > 0 and stripped[:num_end].isdigit():
                blocks.append(("numbered", stripped))
            else:
                blocks.append(("p", stripped))
        else:
            blocks.append(("p", stripped))

    if code_lines:
        blocks.append(("code", "\n".join(code_lines)))

    return blocks


def build_pdf(markdown_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=4,
    )
    heading1 = ParagraphStyle(
        "Heading1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceBefore=4,
        spaceAfter=8,
    )
    heading2 = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "BulletCustom",
        parent=normal,
        leftIndent=12,
        bulletIndent=0,
        firstLineIndent=0,
    )
    code = ParagraphStyle(
        "CodeBlock",
        fontName="Courier",
        fontSize=8.8,
        leading=11,
        leftIndent=8,
    )

    story = []
    markdown_text = markdown_path.read_text(encoding="utf-8")

    for kind, content in parse_markdown(markdown_text):
        if kind == "h1":
            story.append(Paragraph(content, heading1))
        elif kind == "h2":
            story.append(Paragraph(content, heading2))
        elif kind == "p":
            story.append(Paragraph(content, normal))
        elif kind == "bullet":
            story.append(Paragraph(content, bullet, bulletText="-"))
        elif kind == "numbered":
            story.append(Paragraph(content, normal))
        elif kind == "code":
            story.append(Spacer(1, 2))
            story.append(Preformatted(content, code))
            story.append(Spacer(1, 4))
        elif kind == "blank":
            story.append(Spacer(1, 4))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Prosperity 4 Playbook",
        author="Repository-derived guide",
    )
    doc.build(story)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown_path = repo_root / "docs" / "prosperity4_guide.md"
    pdf_path = repo_root / "docs" / "prosperity4_guide.pdf"

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown source not found: {markdown_path}")

    build_pdf(markdown_path, pdf_path)
    print(f"Generated PDF: {pdf_path}")


if __name__ == "__main__":
    main()

