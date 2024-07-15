from typing import Dict, List
import pymongo

class Pages:
    def __init__(self, manager, lang):
        self.manager = manager
        self.lang = lang
        self.col = self.manager.db.get_collection(f"pages_{self.lang}")

    def addPage(self, title, redirect, chunks, session=None):
        self.col.insert_one({
            "title_official": title,
            "title_normalized": Pages.normalizeTitle(title),
            "redirect": None if not redirect else redirect,
            "chunks": chunks
        }, session=session)

    def getLeafPage(self, title, known=None):
        if known is None: known = []

        if title in known:
            return None, []

        page_routes = [
            # redirect
            self.getLeafPage(page["redirect"], known=known+[page["title_official"]])

            if page.get("redirect") else
            # real page
            (page, known+[page["title_official"]])

            for page in self.getPageCandidates(title)
            if page["title_official"] not in known
        ]

        # filter
        page_routes = [ item for item in page_routes if item[0] is not None ]
        if page_routes:
            # take first hit
            return page_routes[0]
        return None, []

    def getPageCandidates(self, title) -> List[Dict]:
        # try official title
        page = self.col.find_one({"title_official": title}, {"_id": False})

        # direct hit
        if page is not None:
            return [page]

        # check normalized titles
        return list(self.col.find({"title_normalized": Pages.normalizeTitle(title)}, {"_id": False}))

    def addEntityToPage(self, title, qid, session=None):
        page, page_chain = self.getLeafPage(title)

        if page is not None:
            for indirect_page_title in page_chain[1:-1]:
                self.col.update_one(
                    {"title_official": page["title_official"]},
                    { "$addToSet": {"qids_indirect": qid}}
                , session=session)

            # add initial qid page
            self.col.update_one(
                {"title_official": page_chain[0]},
                { "$addToSet": {"qids_source": qid}}
            , session=session)

            # add final qid page
            self.col.update_one(
                {"title_official": page_chain[-1]},
                { "$addToSet": {"qids_target": qid}}
            , session=session)

    def createTitleIndices(self, session=None):
        self.col.create_index([
            ("title_official", pymongo.HASHED),
        ], name="hashed_official_titles", session=session)

        self.col.create_index([
            ("title_normalized", pymongo.HASHED),
        ], name="hashed_normalized_titles", session=session)

    def clearTitleIndices(self, session=None):
        self.col.drop_index("hashed_official_titles", session=session)
        self.col.drop_index("hashed_normalized_titles", session=session)

    @classmethod
    def normalizeTitle(cls, title):
        return title.lower()
