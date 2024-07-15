import pymongo

class Entities:
    def __init__(self, manager):
        self.manager = manager
        self.col = self.manager.db.get_collection("entities")

    def addEntity(self, qid, lang_labels, lang_titles, session=None):
        self.col.insert_one({
            "qid": qid,
            **lang_labels,
            **lang_titles
        }, session=session)

    def createQIDIndex(self, session=None):
        self.col.create_index([
            ("qid", pymongo.HASHED),
        ], name="hashed_qids", session=session)

    def clearQIDIndex(self, session=None):
        self.col.drop_index("hashed_qids", session=session)