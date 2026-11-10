"""지식 베이스 로드와 청킹.

- 섹션(Section): 인용 단위. 본문 = md + 같은 이름의 CSV 표(채점기 load_kb_dir과 같은 방식).
- 청크(Chunk): 검색 단위. md의 `## 소주제` 단위로 나누고, `(표: X.csv)` 자리에 CSV 내용을 넣는다.
  청크 ID(`PY-01-01#2`)와 인용용 섹션 ID(`PY-01-01`)는 구분한다.
- `kb/finance/FIN-01.html`은 FIN-01-01~04의 중복본(값 대조 완료)이므로 색인하지 않는다.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

SECTION_ID = re.compile(r"^(PY|PR|FIN)-\d{2}-\d{2}$")
_TABLE_REF = re.compile(r"\(표:\s*([^)]+?\.csv)\)")


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    heading: str
    text: str


@dataclass
class Section:
    section_id: str
    subject: str
    title: str
    text: str
    chunks: List[Chunk] = field(default_factory=list)


def _split_h2(md: str):
    """`## ` 제목 기준으로 (heading, body) 목록. 첫 `## ` 앞부분은 첫 조각에 붙인다."""
    parts, heading, buf, preface = [], None, [], []
    for line in md.splitlines():
        if line.startswith("## "):
            if heading is not None:
                parts.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif heading is None:
            if not line.startswith("# "):
                preface.append(line)
        else:
            buf.append(line)
    if heading is not None:
        parts.append((heading, "\n".join(buf).strip()))
    if not parts:
        parts = [("", "\n".join(preface).strip())]
    elif any(p.strip() for p in preface):
        h, b = parts[0]
        parts[0] = (h, ("\n".join(preface).strip() + "\n\n" + b).strip())
    return parts


def load_kb(kb_dir: Path) -> Dict[str, Section]:
    sections: Dict[str, Section] = {}
    for md_path in sorted(Path(kb_dir).glob("*/*.md")):
        sid = md_path.stem
        if not SECTION_ID.match(sid):
            continue
        md = md_path.read_text(encoding="utf-8")
        tables = {t.name: t.read_text(encoding="utf-8").strip()
                  for t in sorted(md_path.parent.glob(f"{sid}-table*.csv"))}
        first = md.splitlines()[0] if md else ""
        title = re.sub(rf"^#\s*{re.escape(sid)}\s*", "", first).strip()
        full = "\n\n".join([md, *tables.values()])
        sec = Section(sid, md_path.parent.name, title, full)

        used = set()
        for i, (heading, body) in enumerate(_split_h2(md)):
            def _inline(m):
                name = m.group(1).strip()
                if name in tables:
                    used.add(name)
                    return "\n[표]\n" + tables[name] + "\n"
                return m.group(0)
            body = _TABLE_REF.sub(_inline, body)
            text = f"[{sid}] {title} > {heading}\n{body}".strip()
            sec.chunks.append(Chunk(f"{sid}#{i}", sid, heading, text))
        leftover = [tables[n] for n in tables if n not in used]
        if leftover and sec.chunks:
            sec.chunks[-1].text += "\n[표]\n" + "\n\n".join(leftover)
        sections[sid] = sec
    if not sections:
        raise RuntimeError(f"지식 베이스가 비어 있습니다: {kb_dir}")
    return sections
