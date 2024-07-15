import os, sys, json
from functools import reduce
from typing import List, Tuple
from wiki_mongo.db import DBManager
from wiki_mongo.sparql import getQIDsForQuery

# from https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def determineTitlesForPagedMentions(dbm, lang: str, qids: List[str], match_mode: str):
    titles = []
    match_itm = "qids_sources" if match_mode == "source" else "qids_targets"
    cursor = dbm.pagedmentions(lang).col.find({match_itm: {"$elemMatch": {"$in": qids}}})

    for pagedmention in cursor:
        titles.append(pagedmention["title_official"])
    return titles

def drawFilteredItems(dbm, lang: str, pagetitles: List[str], qid_map: List[Tuple[List[str], str]], keep_negatives: bool, match_mode: str, trf_mode: str):
    for title in pagetitles:
        pm = dbm.pagedmentions(lang).col.find_one({"title_official": title})

        for text, annotations in pm.get("chunks", []):
            filtered_annotations = []
            filtered_negatives = []
            for (start, stop, ref_title, tgt_ids, src_ids) in annotations:
                # let check every label class
                for qids, lbl in qid_map:
                    ann_qids = src_ids if match_mode == "source" else tgt_ids
                    matched_qids = [ match_qid for match_qid in ann_qids if match_qid in qids ]
                    # store annotation if a match was found
                    if matched_qids:
                        shared_qid = matched_qids if trf_mode == "source" else tgt_ids

                        filtered_annotations.append(
                            (start, stop, lbl, shared_qid)
                        )
                    else:
                        filtered_negatives.append(
                            (start, stop, None)
                        )

            if filtered_annotations:
                yield (text, filtered_annotations, filtered_negatives) if keep_negatives else (text, filtered_annotations)


def generate_dataset(querymap: str, output_file: str, lang: str, keep_negatives: bool, match_qid: str, to_qid: str, max_size: int, split_resultset: int, max_sparql_items: int, sparql_endpoint: str):
    dbm = DBManager()

    if os.path.exists(querymap):
        with open(querymap, "r") as f:
            qm = json.load(querymap)
    else:
        qm = json.loads(querymap)

    assert isinstance(qm, list), f"Querymap is not a list"
    assert all([ isinstance(k, str) and (isinstance(v, str) or v is None) for k,v in qm ]), f"Querymap contains an invalid item"

    qid_map: List[Tuple[List[str], str]] = [
        (getQIDsForQuery(sparqlquery, sparql_endpoint), labelclass)
        for sparqlquery, labelclass in qm
    ]

    print(f"Found {len(qid_map)} queries to perform...", file=sys.stderr)

    qids_of_interest = list(set(reduce(lambda x,y: x+y, [ qids for qids, _ in qid_map], [])))

    if max_sparql_items > 0 and len(qids_of_interest) > max_sparql_items:
        print(f"Found {len(qids_of_interest)} items but only {max_sparql_items} were allowed. Stopping.", file=sys.stderr)
        sys.exit(-1)

    pagetitles_of_interest = set()

    if split_resultset > 0:
        print(f"Using chunked QID search for {len(qids_of_interest)} QIDs...", file=sys.stderr)
        for qids_subset in divide_chunks(qids_of_interest, split_resultset):
            pagetitles_of_interest.update(determineTitlesForPagedMentions(dbm, lang, qids_subset, match_qid))
    else:
        print(f"Using non-chunked QID search for {len(qids_of_interest)} QIDs...", file=sys.stderr)
        pagetitles_of_interest.update(determineTitlesForPagedMentions(dbm, lang, qids_of_interest, match_qid))

    print(f"Found {len(pagetitles_of_interest)} pages...", file=sys.stderr)
    print(f"Extracting the sentences...", file=sys.stderr)
    n_samples = 0
    with open(output_file, "w") as f:
        sentence_sample_generator = drawFilteredItems(dbm, lang, pagetitles_of_interest, qid_map, keep_negatives, match_qid, to_qid)
        for sentence_sample in sentence_sample_generator:
            # define early stopping condition
            if max_size > 0 and n_samples >= max_size:
                print(f"Stop after {n_samples} due to explicit limit.", file=sys.stderr)
                break
            n_samples += 1

            # write to JSONL file
            sample = {
                "text": sentence_sample[0], "label": sentence_sample[1],
            }
            if keep_negatives:
                sample["neg_label"] = sentence_sample[2]

            f.write(json.dumps(sample)+"\n")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("querymap", type=str, help="Which [{query results} -> label class] mentions should be extracted. If no mapping value is provided, keep qid. The queries should start with 'SELECT ?item WHERE'")
    parser.add_argument("output_file", type=str, help="Output file, encoded as jsonl")
    parser.add_argument("-m", "--match-qid", default="source", choices=["source", "target"], type=str, help="Which match type should be used to identify mentions.")
    parser.add_argument("-t", "--to-qid", default="source", choices=["source", "target"], type=str, help="Which match type should be used in output file, if no label class is given.")
    parser.add_argument("-l", "--language", default='en', type=str, help="Wikipedia language code.")
    parser.add_argument("-n", "--size", default=-1, type=int, help="Maximum number of extracted sentences")
    parser.add_argument('--keep-negatives', action='store_true')
    parser.add_argument("--split-resultset", default=-1, type=int, help="Maximum number of query result items to process at each mongodb query")
    parser.add_argument("--max-sparql-items", default=-1, type=int, help="Maximum number of allowed query result items")
    parser.add_argument("-e", "--sparql-endpoint", default='https://query.wikidata.org/sparql', type=str, help="SPARQL WikiData endpoint")

    args = parser.parse_args()

    generate_dataset(args.querymap, args.output_file, args.language, args.keep_negatives, args.match_qid, args.to_qid, args.size, args.split_resultset, args.max_sparql_items, args.sparql_endpoint)