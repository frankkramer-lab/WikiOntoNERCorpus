import os, sys, json
from multiprocessing import Process
from typing import List
from wiki_mongo.db import DBManager
from wiki_mongo.wtfwiki import WtfWikiManager
from wiki_mongo.parser.Page import Page
from wiki_mongo.parser.Entity import Entity

def importPages(dbmgr, wwmgr, lang: str, input_file: str, n_workers: int = 1):
    def processPages(lang: str, input_file: str, n_workers: int = 1, worker_idx: int = 0, dbm_cfg = None, wwm_cfg = None, increase_ports=False):
        if dbm_cfg is None: dbm_cfg = {}
        if isinstance(dbm_cfg, str): dbm_cfg = json.loads(dbm_cfg)
        dbm = DBManager.fromConfig(dbm_cfg)

        if wwm_cfg is None: wwm_cfg = {}
        if isinstance(wwm_cfg, str): wwm_cfg = json.loads(wwm_cfg)
        wwm = WtfWikiManager.fromConfig(wwm_cfg)

        if increase_ports: wwm.port += worker_idx

        dbm_pages = dbm.pages(lang)
        for page in Page.load_from_dumpfile(input_file, offset=worker_idx, skip_after=n_workers):
            if page is not None:
                dbm_pages.addPage(
                    page.title,
                    page.redirect,
                    wwm.parse(page.text) if page.text else None
                )

    if n_workers == 1:
        # takes ~190 pages/sec
        processPages(lang, input_file, n_workers=1, worker_idx=0, dbm_cfg=dbmgr.getConfig(), wwm_cfg=wwmgr.getConfig())
    else:
        processes = [ Process(
            target=processPages,
            args=(lang, input_file),
            kwargs={
                "n_workers": n_workers,
                "worker_idx": p_idx,
                "dbm_cfg": dbmgr.getConfig(),
                "wwm_cfg": wwmgr.getConfig(),
                "increase_ports": True
            }
        ) for p_idx in range(n_workers) ]
        for p in processes: p.start()
        for p in processes: p.join()

    print(f"Pages have been imported... Creating title indices now...", file=sys.stderr)
    dbmgr.pages(lang).createTitleIndices()


def importEntities(dbmgr, langs: List[str], input_file: str, n_workers: int = 1):
    def processEntities(langs: str, input_file: str, n_workers: int = 1, worker_idx: int = 0, dbm_cfg = None):
        if dbm_cfg is None: dbm_cfg = {}
        if isinstance(dbm_cfg, str): wwm_cfg = json.loads(dbm_cfg)
        dbm = DBManager.fromConfig(dbm_cfg)

        dbm_entities = dbm.entities()
        for entity in Entity.load_from_dumpfile(input_file, langs, offset=worker_idx, skip_after=n_workers):
            if entity is not None:
                dbm_entities.addEntity(
                    entity.qid,
                    entity.lang_labels,
                    entity.lang_titles
                )

    if n_workers == 1:
        # run in same process
        processEntities(langs, input_file, n_workers=1, worker_idx=0, dbm_cfg=dbmgr.getConfig())
    else:
        processes = [ Process(
            target=processEntities,
            args=(langs, input_file),
            kwargs={
                "n_workers": n_workers,
                "worker_idx": p_idx,
                "dbm_cfg": dbmgr.getConfig()
            }
        ) for p_idx in range(n_workers) ]
        for p in processes: p.start()
        for p in processes: p.join()
    dbmgr.entities().createQIDIndex()

def importDump(dump_type: str, wiki_lang: str, input_file: str, n_workers: int = 1):
    if dump_type in ["entities"] and wiki_lang:
        print(f"Loading {dump_type} which are language-agnostic...", file=sys.stderr)

    if not os.path.exists(input_file):
        raise Exception(f"File {input_file} not found.")

    dbm = DBManager()
    wwm = WtfWikiManager()
    if dump_type == "pages":
        importPages(dbm, wwm, wiki_lang, input_file, n_workers=n_workers)

    elif dump_type == "entities":
        importEntities(dbm, wiki_lang.split(","), input_file, n_workers=n_workers)
    else:
        raise Exception(f"Unknown type: {dump_type}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file to load")
    parser.add_argument("-t", "--type", default="pages", choices=["pages", "entities"], type=str, help="Which type of dump, e.g. pages or entities")
    parser.add_argument("-l", "--language", default='en', type=str, help="Wikipedia language code. If type is entities, provide multiple comma separated codes, e.g. 'de,en,es'")
    parser.add_argument("-n", "--n_workers", default=1, type=int, help="Number of workers to process input")
    args = parser.parse_args()

    importDump(args.type, args.language, args.input_file, n_workers=args.n_workers)