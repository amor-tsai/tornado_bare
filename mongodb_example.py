#!/usr/bin/python


from pymongo import MongoClient

client = MongoClient('mongodb://127.0.0.1:8000/',serverSelectionTimeoutMS=50)

db=client.rnndatabase
collect1 = db.queries

for document in collect1.find():
	print(document)