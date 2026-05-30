import re


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags(value: str) -> str:
    return _HTML_TAG_RE.sub("", value)
