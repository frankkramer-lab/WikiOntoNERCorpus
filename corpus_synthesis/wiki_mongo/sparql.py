import requests, re

QID_PTN = re.compile(r'.*(?P<qid>Q[0-9]+)$')

def getQIDsForQuery(query, endpoint, qid_column="item"):
    params = {"query": query }
    headers = {"Accept": "application/sparql-results+json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    obj = response.json()

    assert qid_column in obj.get("head", {}).get("vars", [])
    result_lines = obj.get("results", {}).get("bindings", [])

    qids = []
    for rl in result_lines:
        qid_uri = rl.get(qid_column, {}).get("value", "")
        m = QID_PTN.fullmatch(qid_uri)
        if m:
            qids.append(m.group("qid"))

    return qids