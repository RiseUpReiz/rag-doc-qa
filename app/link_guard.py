"""Link guard: remove links and email addresses that point outside approved domains.

Runs on the model's answer after generation, so even if a poisoned document talks the model
into repeating a phishing link, the link never reaches the user. Default-deny: a host is
allowed only if it equals an allowed domain or is a subdomain of one.
"""
import re
from urllib.parse import urlsplit

PLACEHOLDER = "[unverified link removed]"
NOTE = ("Note: one or more links or email addresses were removed because they point to "
        "unapproved domains.")

_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
# The last label must start with a letter and have 2+ characters, so 0.70, 3.6 and "e.g" don't match.
_DOMAIN = rf"(?:{_LABEL}\.)+[A-Za-z][A-Za-z0-9-]*[A-Za-z0-9]"
_URL_CHARS = r"[^\s<>()\[\]{}\"'`]"

_MARKDOWN = rf"\[(?P<md_text>[^\]\n]*)\]\((?P<md_target>[^)\s]+)\)"
_URL = rf"(?P<url>https?://{_URL_CHARS}+)"
_EMAIL = rf"(?<![\w.%+-])(?P<email>[A-Za-z0-9._%+-]+@(?P<email_domain>{_DOMAIN}))(?![\w-])"
# Not preceded by a word character, "." "@" "/" or "-", so "handbook.md" inside
# "halden_handbook.md" (or a domain inside a URL path) is never matched on its own.
_BARE = rf"(?<![\w.@/-])(?P<bare>{_DOMAIN}(?:/{_URL_CHARS}*)?)(?![\w-])"

_LINK = re.compile("|".join([_MARKDOWN, _URL, _EMAIL, _BARE]), re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"*"


def _split_trailing(text: str) -> tuple[str, str]:
    """Separate sentence punctuation that ended up on the end of a URL or domain."""
    stripped = text.rstrip(_TRAILING_PUNCTUATION)
    return stripped, text[len(stripped):]


def is_allowed(host: str | None, allowed_domains: list[str]) -> bool:
    """True if host equals an allowed domain or is a subdomain of one. Empty allow-list allows nothing."""
    if not host:
        return False
    host = host.lower().rstrip(".")
    for domain in allowed_domains:
        domain = domain.lower().strip(".")
        if domain and (host == domain or host.endswith("." + domain)):
            return True
    return False


def _target_host(target: str) -> tuple[bool, str | None]:
    """For a link target, return (is_external, host). Relative links like "#notes" aren't external."""
    if target.lower().startswith("mailto:"):
        return True, target.split("@", 1)[-1].split("?", 1)[0]
    if "://" in target:
        return True, urlsplit(target).hostname
    if "@" in target:
        return True, target.split("@", 1)[1]
    match = re.fullmatch(rf"({_DOMAIN})(?:/.*)?", target)
    return (True, match.group(1)) if match else (False, None)


def _is_link(text: str) -> bool:
    match = _LINK.fullmatch(text.strip())
    return bool(match) and match.group("md_target") is None


def guard(answer: str, allowed_domains: list[str], source_names: list[str]) -> tuple[str, list[str]]:
    """Replace links, emails and bare domains outside allowed_domains with a placeholder.

    Returns the cleaned answer and the removed items, in order. source_names (the retrieved
    documents' file names) are never treated as domains. If anything was removed, a note is
    appended on its own line.
    """
    sources = set(source_names)
    removed: list[str] = []

    def replace(match: re.Match) -> str:
        if match.group("md_target") is not None:
            text, target = match.group("md_text"), match.group("md_target")
            external, host = _target_host(target)
            if not external or is_allowed(host, allowed_domains):
                return f"[{_LINK.sub(replace, text)}]({target})"
            if _is_link(text):
                removed.append(target)
                return PLACEHOLDER
            kept_text = _LINK.sub(replace, text)
            removed.append(target)
            return f"{kept_text} {PLACEHOLDER}"

        if match.group("email") is not None:
            if is_allowed(match.group("email_domain"), allowed_domains):
                return match.group(0)
            removed.append(match.group("email"))
            return PLACEHOLDER

        raw = match.group("url") or match.group("bare")
        link, trailing = _split_trailing(raw)
        if match.group("url") is not None:
            host = urlsplit(link).hostname
        else:
            if link in sources:
                return raw
            host = link.split("/", 1)[0]
        if is_allowed(host, allowed_domains):
            return raw
        removed.append(link)
        return PLACEHOLDER + trailing

    cleaned = _LINK.sub(replace, answer)
    if removed:
        cleaned = cleaned.rstrip() + "\n\n" + NOTE
    return cleaned, removed
