import os, time
from typing import Dict
import requests
from functools import reduce

DEFAULT_HOST = os.environ.get("WTFWIKI_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("WTFWIKI_PORT", "3000"))

def unnested(item):
    if isinstance(item, dict):
        return [item]
    elif isinstance(item, list):
        return reduce(lambda x,y: x+y, [
            unnested(e) for e in item
        ], [])
    return []

def extractItem(item):
    chunks = []
    if "title" in item and item.get("item"):
        chunks.append((item.get("item"), []))

    if "text" in item:
        # text node
        text = item["text"]
        linkItems = []
        for link in item.get("links", []):
            # skip non-internal links
            if link.get("type", None) != "internal":
                continue

            # look for spans
            link_text = link.get("text", link["page"])
            try: idx = text.index(link_text)
            except ValueError: idx = None

            if idx is None:
                try: idx = text.lower().index(link_text.lower())
                except ValueError: idx = None

            # skip if text was not found
            if idx is None:
                continue

            linkItems.append((idx, idx+len(link_text), link["page"]))
        chunks.append((text, linkItems))

    if "sections" in item:
        chunks += reduce(lambda x,y: x+y, [ extractItem(section) for section in item.get("sections", [])], [])

    if "paragraphs" in item:
        chunks += reduce(lambda x,y: x+y, [ extractItem(paragraph) for paragraph in item.get("paragraphs", [])], [])

    if "sentences" in item:
        chunks += reduce(lambda x,y: x+y, [ extractItem(sentence) for sentence in item.get("sentences", [])], [])

    if "lists" in item:
        lists = item.get("lists", [])
        lists_items = unnested(lists)
        chunks += reduce(lambda x,y: x+y, [ extractItem(item) for item in lists_items ], [])

    return chunks

class WtfWikiManager:
    def __init__(self, host: str = None, port: int = None):
        self.host = DEFAULT_HOST if host is None else host
        self.port = DEFAULT_PORT if port is None else port

    def getConfig(self):
        return {
            "host": self.host,
            "port": self.port
        }

    @classmethod
    def fromConfig(cls, cfg):
        return WtfWikiManager(**cfg)

    def parse(self, wikitext: str, n_retrys: int = 3, retry_cooldown: float = 0.5, extract=True) -> Dict:
        wtf_url = f"http://{self.host}:{self.port}/toJSON"
        result = None
        for _ in range(n_retrys):
            try:
                response = requests.post(wtf_url, json={"text": wikitext})
                response.raise_for_status()
                result = response.json()
                break
            except:
                time.sleep(retry_cooldown)
                continue

        if result is None or not result.get("success", False):
            import json
            with open("./wikitext_errors.jsonl", "a") as f:
                f.write(json.dumps({"text": wikitext})+"\n")
            return None

        doc = result["result"]
        if extract:
            return extractItem(doc)
        else:
            return doc
