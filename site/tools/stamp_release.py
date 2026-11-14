import argparse
import html as html_lib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import urljoin

parser = argparse.ArgumentParser()
parser.add_argument("--pyproject", required=True, type=Path)
parser.add_argument("--changelog", required=True, type=Path)
parser.add_argument("--base-url", required=True)
parser.add_argument("pages", nargs="+", type=Path)
args = parser.parse_args()
version = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))["project"]["version"]
changelog = args.changelog.read_text(encoding="utf-8")
heading = re.search(rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE)
if not heading:
    raise SystemExit(f"{args.changelog} 沒有 {version} 的發布日期")
release = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def releases():
    marks = list(release.finditer(changelog))
    for i, m in enumerate(marks):
        body = changelog[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(changelog)]
        counts = {}
        for section in re.finditer(r"^### (\w+)\n(.*?)(?=^### |\Z)", body, re.MULTILINE | re.DOTALL):
            counts[section.group(1)] = len(re.findall(r"^- ", section.group(2), re.MULTILINE))
        yield m.group(1), m.group(2), m.group(0)[3:], counts


def anchor(title):
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def changelog_items(tag):
    attrs = {k: a or b for k, a, b in re.findall(r'\b(data-[\w-]+)=(?:"([^"]*)"|\'([^\']*)\')', tag)}
    labels = json.loads(html_lib.unescape(attrs["data-labels"]))
    items = []
    for ver, date, title, counts in list(releases())[: int(attrs["data-count"])]:
        missing = sorted(set(counts) - set(labels))
        if missing:
            raise SystemExit(f"{args.changelog} 的 {ver} 有 data-labels 沒列的分類：{'、'.join(missing)}")
        parts = "".join(f"<span>{labels[k]} {n} {attrs['data-unit']}</span>" for k, n in counts.items() if n)
        href = f"{attrs['data-href']}#{anchor(title)}"
        items.append(f'<li><a href="{href}"><b>v{ver}</b><time datetime="{date}">{date}</time>{parts}</a></li>')
    return "".join(items)


base = args.base_url.rstrip("/") + "/"
for page in args.pages:
    html = page.read_text(encoding="utf-8")
    for key, text in {"version": f"v{version}", "date": heading.group(1)}.items():
        pattern = re.compile(rf'(<(\w+)[^>]*\bdata-release="{key}"[^>]*>)(</\2>)')
        html, count = pattern.subn(lambda m, text=text: m.group(1) + text + m.group(3), html)
        if count != 1:
            raise SystemExit(f'{page} 應該剛好有一個空的 data-release="{key}"，實際有 {count} 個')
    html, filled = re.subn(r"(<ol\b[^>]*\bdata-changelog\b[^>]*>)(</ol>)", lambda m: m.group(1) + changelog_items(m.group(1)) + m.group(2), html)
    if filled > 1:
        raise SystemExit(f"{page} 最多只能有一個空的 data-changelog，實際有 {filled} 個")
    html, count = re.subn(
        r"<(?:link|meta|base)\b[^>]*\bdata-absolute\b[^>]*>",
        lambda tag: re.sub(
            r'\b(href|content)="([^"]*)"',
            lambda attr: f'{attr.group(1)}="{urljoin(base, attr.group(2))}"',
            tag.group(0),
            count=1,
        ),
        html,
    )
    if not count:
        raise SystemExit(f"{page} 沒有標 data-absolute 的網址")
    page.write_text(html, encoding="utf-8")
    print(f"{page}: v{version}, {heading.group(1)}, {count} 個網址以 {base} 為基準，最近更新 {filled} 處")
