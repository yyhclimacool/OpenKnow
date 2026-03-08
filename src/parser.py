from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import VisionConfig

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MARKDOWN_IMAGE_RE = re.compile(r"(!\[([^\]]*)\]\(([^)]+)\))")


@dataclass
class ParsedDocument:
    source_path: str
    content: str
    title: str = ""
    doc_type: str = "markdown"
    metadata: dict = field(default_factory=dict)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _describe_image(image_path: Path, vision_cfg: VisionConfig, alt_text: str = "") -> str:
    """Use Vision API to generate a text description of an image."""
    from openai import OpenAI

    client = OpenAI()
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    suffix = image_path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"

    context = f'图片的 alt 文字为："{alt_text}"。' if alt_text else ""
    prompt = (
        f"{context}请用中文详细描述这张图片的内容。"
        "如果图片包含文字，请完整提取。"
        "如果是流程图或架构图，请描述其结构和关系。"
    )

    try:
        resp = client.chat.completions.create(
            model=vision_cfg.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        logger.exception("Failed to describe image: %s", image_path)
        return ""


def _expand_images(md_content: str, md_path: Path, vision_cfg: VisionConfig) -> str:
    """Replace inline image references with Vision API descriptions."""

    def replace_match(m: re.Match) -> str:
        _full, alt, src = m.group(1), m.group(2), m.group(3)

        if src.startswith(("http://", "https://")):
            logger.debug("Skipping remote image: %s", src)
            return _full

        img_path = (md_path.parent / src).resolve()
        if not img_path.exists() or not _is_image(img_path):
            logger.debug("Image not found or unsupported: %s", img_path)
            return _full

        logger.info("Describing image via Vision API: %s (alt=%r)", img_path.name, alt)
        description = _describe_image(img_path, vision_cfg, alt_text=alt)
        if not description:
            return _full

        label = alt or img_path.stem
        return f"[图片：{label}]\n{description}"

    return MARKDOWN_IMAGE_RE.sub(replace_match, md_content)


def parse_markdown(
    path: Path, vision_cfg: VisionConfig | None = None
) -> ParsedDocument:
    """Parse a markdown file, expanding inline images via Vision API if configured."""
    text = path.read_text(encoding="utf-8")

    if vision_cfg and MARKDOWN_IMAGE_RE.search(text):
        text = _expand_images(text, path, vision_cfg)

    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    return ParsedDocument(
        source_path=str(path),
        content=text,
        title=title,
        doc_type="markdown",
    )


def scan_directory(
    directory: Path, vision_cfg: VisionConfig
) -> list[ParsedDocument]:
    """Scan a directory recursively, parsing all markdown files.

    Images referenced inline in markdown are described via Vision API and
    their descriptions are embedded into the document content.
    """
    docs: list[ParsedDocument] = []

    md_files = sorted(directory.rglob("*.md"))
    logger.info("Found %d markdown files in %s", len(md_files), directory)
    for md_path in md_files:
        logger.info("Parsing: %s", md_path.relative_to(directory) if md_path.is_relative_to(directory) else md_path)
        doc = parse_markdown(md_path, vision_cfg=vision_cfg)
        docs.append(doc)
        logger.debug("  -> title=%r, %d chars", doc.title or "(none)", len(doc.content))

    logger.info("Scanned %s: %d markdown files", directory, len(docs))
    return docs
