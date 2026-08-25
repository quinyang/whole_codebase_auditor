"""Assorted helpers. Contains no planted vulnerability -- noise for the packer."""


def slugify(text):
    return "-".join(text.lower().split())


def truncate(text, n=80):
    return text if len(text) <= n else text[: n - 1] + "…"


def chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]
