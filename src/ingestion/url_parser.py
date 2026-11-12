import re
from typing import List, Optional
import requests
from langchain_core.documents import Document

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

class URLArticleParser:
    """Fetches and parses clean text from news articles, blogs, and web links."""

    @classmethod
    def parse_url(cls, url: str, timeout: int = 15) -> Document:
        """Downloads a web page and extracts its title and primary article text."""
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            response.raise_for_status()
            html_text = response.text
        except Exception as e:
            raise RuntimeError(f"Failed to fetch content from '{url}': {e}")

        title = url
        article_text = ""

        if HAS_BS4:
            soup = BeautifulSoup(html_text, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "noscript"]):
                tag.decompose()

            # Extract Title
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text().strip()

            # Extract main article or paragraph contents
            main_content = soup.find("article") or soup.find("main") or soup.find("body")
            if main_content:
                paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "li"])
                text_blocks = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
                article_text = "\n\n".join(text_blocks)
            else:
                article_text = soup.get_text(separator="\n", strip=True)
        else:
            # Fallback regex-based extraction if BeautifulSoup is not present
            clean_text = re.sub(r"<(script|style|nav|footer|header).*?</\1>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
            title_match = re.search(r"<title>(.*?)</title>", clean_text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

            # Extract paragraph tags
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", clean_text, flags=re.DOTALL | re.IGNORECASE)
            clean_paragraphs = []
            for p in paragraphs:
                p_clean = re.sub(r"<[^>]+>", "", p).strip()
                if len(p_clean) > 30:
                    clean_paragraphs.append(p_clean)
            article_text = "\n\n".join(clean_paragraphs)

        if not article_text or len(article_text.strip()) < 50:
            raise ValueError(f"Could not extract sufficient text content from URL: {url}")

        content_with_header = f"Article Title: {title}\nSource URL: {url}\n\n{article_text}"

        return Document(
            page_content=content_with_header,
            metadata={
                "source": url,
                "title": title,
                "type": "web_article",
                "page": 1,
            }
        )

    @classmethod
    def parse_urls(cls, urls: List[str]) -> List[Document]:
        """Parses multiple URLs, skipping any failed links with informative logs."""
        docs = []
        for url in urls:
            url_clean = url.strip()
            if not url_clean:
                continue
            try:
                doc = cls.parse_url(url_clean)
                docs.append(doc)
            except Exception as e:
                print(f"[Warning] Skipping '{url_clean}': {e}")
        return docs
