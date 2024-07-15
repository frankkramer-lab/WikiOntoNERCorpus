# Synthesis of Weakly-annotated corpus

This section provides the code to handle several aspects:
- [Once] Downloading and parsing of Wikipedia and WikiData dumps from the official server.
- [Once] Parsing and linking of Wikipedia pages and WikiData entities.
- [Repeated] Synthesis of a weakly-annotated text corpus defined by a WikiData SPARQL query. The query is handled by the [official endpoint](https://query.wikidata.org/).

**Note**: If you just want to experiment without runnnig the entire pipeline, use our web app to generate an weakly-annotated corpus. (The WikiData and Wikipedia dumps were downloaded in Feburary 2024.)
Our web app is available at: [https://ontowiki.misit-augsburg.de/](https://ontowiki.misit-augsburg.de/)

<kbd><img src="https://github.com/frankkramer-lab/WikiOntoNERCorpus/blob/main/assets/OntoCorpus_Screenshot.png" width="600"></kbd>

## How to use
### [Once] Data Preparation
1. Setup recent versions of `docker`, `docker compose` and become part of the `docker` group, e.g. using `sudo usermod -aG docker $USER`. Running `docker ps` should be working. You may need to reboot or log-in again after the `usermod` command in order to succeed.
2. A modern version of Python is required. (e.g. >= 3.10)
3. Increase `vm.max_map_count` to `1677720` (for MongoDB) by running `sudo su -c 'echo "vm.max_map_count=1677720" >> /etc/sysctl.conf' && sudo sysctl -p`. \
   Update the MongoDB cache size `wiredTigerCacheSizeGB` to a plausible value (to avoid OOM issues).
4. Prepare the database by using the following script / commands. Change certain values according to your needs and hardware. If you run into OOM issues, reduce the `N_WORKERS` variable. Skip the commands for certain languages if you don't need them. 
<details>
<summary>Preparation Script</summary>

```bash
# Define vars
export N_WORKERS=8

# Build image & download images
docker compose build --pull
docker compose pull --ignore-buildable

# Prepare dependencies
python3 -m venv wiki_env
source wiki_env/bin/activate
python3 -m pip install -r requirements.txt

# Download Wikipedia dumps (add or remove a certain line, if desired)
PYTHONPATH=. python3 wiki_mongo/dump_loader.py wiki_dumps/ -t pages -l de
PYTHONPATH=. python3 wiki_mongo/dump_loader.py wiki_dumps/ -t pages -l en
PYTHONPATH=. python3 wiki_mongo/dump_loader.py wiki_dumps/ -t pages -l es
PYTHONPATH=. python3 wiki_mongo/dump_loader.py wiki_dumps/ -t pages -l fr

# Download WikiData dump
PYTHONPATH=. python3 wiki_mongo/dump_loader.py wiki_dumps/ -t entities

# Start containers
docker compose up -d

# Wait till the containers are available...
sleep 10

# Determine IP addresses of containers (to avoid port conflicts due to port forwardings...)
export MONGO_HOST="$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' wiki_database )"
export WTFWIKI_HOST="$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' wiki_parser )"

# Load Wkipedia dumps into DB
PYTHONPATH=. python3 wiki_mongo/dump_parser.py wiki_dumps/dewiki-latest-pages-meta-current.xml.bz2 -l de -t pages -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_parser.py wiki_dumps/enwiki-latest-pages-meta-current.xml.bz2 -l en -t pages -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_parser.py wiki_dumps/eswiki-latest-pages-meta-current.xml.bz2 -l es -t pages -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_parser.py wiki_dumps/frwiki-latest-pages-meta-current.xml.bz2 -l fr -t pages -n "$N_WORKERS"

# Stop WTFWiki parser
docker compose down wiki-wtf

# Load WikiData dumps into DB along with the list of languages to support
PYTHONPATH=. python3 wiki_mongo/dump_parser.py wiki_dumps/latest-all.json.bz2 -t entities -l "de,en,es,fr" -n "$N_WORKERS"

# Link WikiData titles to Wikipedia pages
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a link -l de -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a link -l en -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a link -l es -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a link -l fr -n "$N_WORKERS"

# Extract paged mentions
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a pagedmentions -l de -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a pagedmentions -l en -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a pagedmentions -l es -n "$N_WORKERS"
PYTHONPATH=. python3 wiki_mongo/dump_process.py -a pagedmentions -l fr -n "$N_WORKERS"
```
</details>
Afterwards, the data preparation is done.

### [Repeated] How to generate a new dataset
1. Make sure that the database container is running and the data preparation script succeeded.
2. Run the following commands:
<details>
<summary>Generation Script</summary>

```bash
# Determine IP of DB container
export MONGO_HOST="$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' wiki_database )"

# Define a WikiData query-based set definition
cat > my_dataset.json <<EOF
[
    ["SELECT ?item WHERE { ?item wdt:P2176 ?something }", "TREATABLE_HEALTH_ISSUE"]
]
EOF

# Generate language-dependent, annotated corpus
PYTHONPATH=. python3 wiki_mongo/generate_dataset.py my_dataset.json corpus_de.jsonl -d de
PYTHONPATH=. python3 wiki_mongo/generate_dataset.py my_dataset.json corpus_en.jsonl -d en
PYTHONPATH=. python3 wiki_mongo/generate_dataset.py my_dataset.json corpus_es.jsonl -d es
PYTHONPATH=. python3 wiki_mongo/generate_dataset.py my_dataset.json corpus_fr.jsonl -d fr
```
</details>
You can repeat these steps for other SPARQL queries as well.

Congratz! Have fun with your custom corpus.
