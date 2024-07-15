import sys
import bz2, json
from typing import Dict, List


class Entity:
    def __init__(self, qid: str, lang_labels: Dict[str, str], lang_titles: Dict[str, str]):
        self.qid: str = qid
        self.lang_labels: Dict[str, str] = lang_labels
        self.lang_titles: Dict[str, str] = lang_titles

    @classmethod
    def from_jsonl(cls, jsonl: Dict, langs: List[str]):
        if jsonl.get("type", None) != "item":
            return None

        qid = jsonl.get("id")
        if not qid: return None

        labels = jsonl.get("labels", {})
        sitelinks = jsonl.get("sitelinks", {})

        lang_labels = {
            f"label_{lang}": labels[lang].get("value")
            for lang in langs
            if lang in labels
        }
        lang_titles = {
            f"title_{lang}": sitelinks[f"{lang}wiki"].get("title")
            for lang in langs
            if f"{lang}wiki" in sitelinks
        }
        return Entity(qid, lang_labels, lang_titles)

    def __repr__(self):
        if self.lang_labels:
            t_lang, t_val = list(self.lang_labels.items())[0]
            fmt_title = f" [{t_lang}: {repr(t_val)}]"
        else: fmt_title = ""
        return f"<Entity {self.qid}{fmt_title}>"

    @classmethod
    def load_from_dumpfile(cls, dumpfile: str, langs: List[str], offset: int = 0, skip_after: int = 1):
        with bz2.open(dumpfile, "rt") as f:
            n_skipping = offset

            for i_line, line in enumerate(f):
                if line.startswith("[") or line.startswith("]"):
                    continue

                if not line.startswith("{"):
                    print(f"Line at {i_line} has unexpected content", file=sys.stderr)
                    continue

                if n_skipping == 0:
                    line = line.strip()
                    # remove trailing comma
                    if line.endswith(","):
                        line = line[:-1]

                    entity = Entity.from_jsonl(json.loads(line), langs)
                    yield entity
                    n_skipping = skip_after

                n_skipping -= 1
