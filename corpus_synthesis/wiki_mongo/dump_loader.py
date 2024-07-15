import os, sys, json
from tqdm import tqdm
import requests

def downloadDump(dump_type: str, wiki_lang: str, output_dir: str, overwrite: bool = False):
    if dump_type in ["entities"] and wiki_lang:
        print("Downloading {dump_type} which are language-agnostic...", file=sys.stderr)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exists_ok=True)

    dump_files = {
        "entities": ("https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2", "latest-all.json.bz2"),
        "pages": (f"https://dumps.wikimedia.org/{wiki_lang}wiki/latest/{wiki_lang}wiki-latest-pages-meta-current.xml.bz2", f"{wiki_lang}wiki-latest-pages-meta-current.xml.bz2")
    }

    file_url, file_name = dump_files[dump_type]
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path) and not overwrite:
        print(f"File {file_name} already exists.", file=sys.stderr)
        return

    print(f"Try to download file from '{file_url}' to '{file_path}'...", file=sys.stderr)
    # download according to: https://stackoverflow.com/a/10744565
    bs = 4096
    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    try:
        with open(file_path + "_headers.txt", "w") as h:
            json.dump({**response.headers}, h)

        try:
            expected_size = int(response.headers.get("Content-Length", "0"))
        except:
            expected_size = 0

        with open(file_path, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=bs), total=expected_size//bs):
                handle.write(data)
    except:
        # Cleanup files if an error occurred
        os.remove(file_path)
        os.remove(file_path + "_headers.txt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", default=".", type=str, help="directory to store the files")
    parser.add_argument("-t", "--type", default="pages", choices=["pages", "entities"], type=str, help="Which type of dump, e.g. pages or entities")
    parser.add_argument("-l", "--language", default='en', type=str, help="Wikipedia language code")
    parser.add_argument("-f", "--overwrite", action="store_true")
    args = parser.parse_args()

    # run evaluation...
    downloadDump(args.type, args.language, args.directory, overwrite=args.overwrite)