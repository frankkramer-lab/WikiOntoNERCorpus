import pymongo

class PagedMentions:
    def __init__(self, manager, lang):
        self.manager = manager
        self.lang = lang
        self.col = self.manager.db.get_collection(f"pagedmentions_{self.lang}")

    def addPagedMentions(self, title, chunks, qids_sources, qids_targets, session=None):
        self.col.insert_one({
            "title_official": title,
            "chunks": chunks,
            "qids_sources": qids_sources,
            "qids_targets": qids_targets
        }, session=session)


    def createQIDMentionsIndices(self, session=None):
        self.col.create_index([
            ("qids_sources", pymongo.ASCENDING),
        ], name="asc_mentions_sources", session=session)

        self.col.create_index([
            ("qids_targets", pymongo.ASCENDING),
        ], name="asc_mentions_targets", session=session)

        self.col.create_index([
            ("title_official", pymongo.HASHED),
        ], name="hashed_titles", session=session)

    def clearQIDMentionsIndices(self, session=None):
        self.col.drop_index("asc_mentions_sources", session=session)
        self.col.drop_index("asc_mentions_targets", session=session)
        self.col.drop_index("hashed_titles", session=session)
