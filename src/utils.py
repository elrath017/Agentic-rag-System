import re

def format_citations(text: str) -> str:
    """
    Transforms raw source citation brackets (e.g. [Source: https://..., Page: 1])
    into an interactive link icon (🔗) or document badge (📄) with hover tooltips
    and direct click-to-open capability.
    """
    if not text:
        return ""

    # 1. Match URLs in citations: [Source: https://... (, Page: X)] or [https://...]
    def url_replacer(match):
        url = match.group(1).strip()
        page = match.group(2)
        page_text = f" | Page {page.strip()}" if page else ""
        tooltip = f"Source: {url}{page_text}"
        return (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'title="{tooltip}" class="citation-link">🔗</a>'
        )

    # Catches:
    # [Source: https://example.com/page, Page: 1]
    # [Source: https://example.com/page]
    # [https://example.com/page]
    # (Source: https://example.com/page, Page: 1)
    text = re.sub(
        r"[\(\[](?:Source:\s*)?(https?://[^\s,\)\]]+)(?:[,\s|]+Page:\s*(\d+))?[\)\]]",
        url_replacer,
        text,
        flags=re.IGNORECASE
    )

    # 2. Match local files: [Source: document.pdf (, Page: X)]
    def doc_replacer(match):
        doc = match.group(1).strip()
        page = match.group(2)
        page_text = f" | Page {page.strip()}" if page else ""
        tooltip = f"Document: {doc}{page_text}"
        return f'<span title="{tooltip}" class="citation-doc">📄</span>'

    text = re.sub(
        r"[\(\[]Source:\s*([^,\)\]]+?\.(?:pdf|txt|md|docx|csv|json))(?:[,\s|]+Page:\s*(\d+))?[\)\]]",
        doc_replacer,
        text,
        flags=re.IGNORECASE
    )

    # 3. Catch any remaining [Source: ...] brackets with URLs or general text
    def generic_source_replacer(match):
        inner = match.group(1).strip()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        first_part = parts[0]
        if first_part.startswith("http://") or first_part.startswith("https://"):
            tooltip = f"Source: {inner}"
            return (
                f'<a href="{first_part}" target="_blank" rel="noopener noreferrer" '
                f'title="{tooltip}" class="citation-link">🔗</a>'
            )
        return f'<span title="Source: {inner}" class="citation-doc">📄</span>'

    text = re.sub(
        r"[\(\[]Source:\s*([^\)\]]+)[\)\]]",
        generic_source_replacer,
        text,
        flags=re.IGNORECASE
    )

    return text

