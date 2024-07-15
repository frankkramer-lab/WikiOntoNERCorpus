import os, pymongo

from wiki_mongo.db import Entities, Pages, PagedMentions

DEFAULT_USERNAME = os.environ.get("MONGO_USERNAME", "wiki_user")
DEFAULT_PASSWORD = os.environ.get("MONGO_PASSWORD", "wiki_pass")
DEFAULT_DATABASE = os.environ.get("MONGO_DATABASE", "wiki_db")
DEFAULT_PORT = int(os.environ.get("MONGO_PORT", "27017"))
DEFAULT_HOST = os.environ.get("MONGO_HOST", "127.0.0.1")

class DBManager:
    def __init__(self, database: str = None, username: str = None, password: str = None, host: str = None, port: int = None):
        self.database_name = DEFAULT_DATABASE if database is None else database
        self.username = DEFAULT_USERNAME if username is None else username
        self.password = DEFAULT_PASSWORD if password is None else password
        self.host = DEFAULT_HOST if host is None else host
        self.port = DEFAULT_PORT if port is None else port

        # add client
        self.client = pymongo.MongoClient(f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}")
        self.db = self.client.get_database(self.database_name)

    def getConfig(self):
        return {
            "database": self.database_name,
            "username": self.username,
            "password": self.password,
            "host": self.host,
            "port": self.port
        }

    @classmethod
    def fromConfig(cls, cfg):
        return DBManager(**cfg)

    def pages(self, lang):
        return Pages(self, lang)

    def entities(self):
        return Entities(self)

    def pagedmentions(self, lang):
        return PagedMentions(self, lang)