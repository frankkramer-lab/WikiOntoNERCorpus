import json
from multiprocessing import Process
from tqdm import tqdm
from wiki_mongo.db import DBManager

def linkPagesWithEntities(dbmgr, lang: str, n_workers: int = 1):
    def link(lang: str, n_workers: int = 1, worker_idx: int = 0, dbm_cfg = None, show_tqdm=False, ordered=True):
        if dbm_cfg is None: dbm_cfg = {}
        if isinstance(dbm_cfg, str): wwm_cfg = json.loads(dbm_cfg)
        dbm = DBManager.fromConfig(dbm_cfg)

        pages = dbm.pages(lang)
        entities = dbm.entities()

        n_skipping = worker_idx
        if show_tqdm:
            assert n_workers == 1, f"Progress bar not supported for multiprocessing implementation."
            n_entities = entities.col.count_documents({})
            pgrs = tqdm(total=n_entities)

        if n_workers > 1:
            assert ordered, f"Unordered configuration selected, but multiprocessing requires ordered iterations"

        with dbm.client.start_session() as session:
            cursor = entities.col.find({}, {"_id": False}, no_cursor_timeout=True, allow_disk_use=True, session=session)
            if ordered:
                cursor = cursor.sort("_id")

            for entity in cursor:
                if n_skipping == 0:
                    # extract qid and language-specific title
                    qid = entity["qid"]
                    lang_title = entity.get(f"title_{lang}", None)

                    if lang_title is not None:
                        # store qid on every page
                        pages.addEntityToPage(lang_title, qid, session=session)

                    n_skipping = n_workers

                n_skipping -= 1

                if show_tqdm:
                    pgrs.update()
        if show_tqdm:
            pgrs.close()

    if n_workers == 1:
        link(lang, n_workers=1, worker_idx=0, dbm_cfg=dbmgr.getConfig(), show_tqdm=True, ordered=True)
    else:
        processes = [ Process(
            target=link,
            args=(lang,),
            kwargs={
                "n_workers": n_workers,
                "worker_idx": p_idx,
                "dbm_cfg": dbmgr.getConfig(),
                "show_tqdm": False,
                "ordered": True
            }
        ) for p_idx in range(n_workers) ]
        for p in processes: p.start()
        for p in processes: p.join()


def extractPagedMentions(dbmgr, lang: str, n_workers: int = 1):
    def extract(lang: str, n_workers: int = 1, worker_idx: int = 0, dbm_cfg = None, show_tqdm=False, ordered=True):
        if dbm_cfg is None: dbm_cfg = {}
        if isinstance(dbm_cfg, str): wwm_cfg = json.loads(dbm_cfg)
        dbm = DBManager.fromConfig(dbm_cfg)

        pages = dbm.pages(lang)
        pagedmentions = dbm.pagedmentions(lang)

        n_skipping = worker_idx
        if show_tqdm:
            assert n_workers == 1, f"Progress bar not supported for multiprocessing implementation."
            n_pages = pages.col.count_documents({})
            pgrs = tqdm(total=n_pages)

        if n_workers > 1:
            assert ordered, f"Unordered configuration selected, but multiprocessing requires ordered iterations"

        with dbm.client.start_session() as session:
            cursor = pages.col.find({}, {"_id": False}, no_cursor_timeout=True, allow_disk_use=True, session=session)
            if ordered:
                cursor = cursor.sort("_id")

            for page in cursor:
                if n_skipping == 0:
                    # only process pages that have chunks
                    chunks = page.get("chunks", None)
                    if chunks:
                        qids_sources = set()
                        qids_targets = set()

                        filtered_chunks = []
                        for text, page_link_items in chunks:
                            filtered_page_links = []
                            for start, stop, page_title in page_link_items:
                                # search for page link
                                try:
                                    linked_page, _ = pages.getLeafPage(page_title)
                                except:
                                    continue

                                if linked_page:
                                    # page has a valid reference to another page
                                    qids_source = linked_page.get("qids_source", [])
                                    qids_target = linked_page.get("qids_target", [])
                                    filtered_page_links.append((start, stop, linked_page["title_official"], qids_source, qids_target))

                                    # update all new mentions on page
                                    qids_sources.update(qids_source)
                                    qids_targets.update(qids_target)

                            filtered_chunks.append(
                                (text, filtered_page_links)
                            )

                        pagedmentions.addPagedMentions(
                            page["title_official"],
                            filtered_chunks,
                            list(qids_sources),
                            list(qids_targets),
                            session=session
                        )

                    n_skipping = n_workers

                n_skipping -= 1

                if show_tqdm:
                    pgrs.update()
        if show_tqdm:
            pgrs.close()

    if n_workers == 1:
        extract(lang, n_workers=1, worker_idx=0, dbm_cfg=dbmgr.getConfig(), show_tqdm=True, ordered=True)
    else:
        processes = [ Process(
            target=extract,
            args=(lang,),
            kwargs={
                "n_workers": n_workers,
                "worker_idx": p_idx,
                "dbm_cfg": dbmgr.getConfig(),
                "show_tqdm": False,
                "ordered": True
            }
        ) for p_idx in range(n_workers) ]
        for p in processes: p.start()
        for p in processes: p.join()

    dbmgr.pagedmentions(lang).createQIDMentionsIndices()

def process(action_type: str, wiki_lang: str, n_workers: int = 1):
    dbm = DBManager()

    if action_type == "link":
        if "," in wiki_lang:
            raise Exception(f"Only one language is supported for action: {action_type}")

        linkPagesWithEntities(dbm, wiki_lang, n_workers=n_workers)
    elif action_type == "pagedmentions":
        if "," in wiki_lang:
            raise Exception(f"Only one language is supported for action: {action_type}")

        extractPagedMentions(dbm, wiki_lang, n_workers=n_workers)

    else:
        raise Exception(f"Unknown action: {action_type}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--action", default="link", choices=["pagedmentions", "link"], type=str, help="Which type of action to apply, e.g. link or pagedmentions")
    parser.add_argument("-l", "--language", default='en', type=str, help="Wikipedia language code. If type is entities, provide multiple comma separated codes, e.g. 'de,en,es'")
    parser.add_argument("-n", "--n_workers", default=1, type=int, help="Number of workers to use")
    args = parser.parse_args()

    process(args.action, args.language, n_workers=args.n_workers)