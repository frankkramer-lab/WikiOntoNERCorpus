#!/usr/bin/env bash

# Download the corpus with the following settings:
# - SPARQL query associated label class 'ATC_CODE':
#   ```
#   # Anything that can be treated by something
#   SELECT ?item
#   WHERE
#   {
#     ?item wdt:P267 ?atccode .
#   }
#   ```
# - Language: German
# - Include negatives: Yes

wget -O ATC_de.jsonl \
  'https://ontocorpus.misit-augsburg.de/download/d19dfb1927ac79480137b1d36abe9449'

# The original ATC corpus can be found here:
# https://ontocorpus.misit-augsburg.de/view?key=66c574b528184d8ebd3bdf1e4b159a4f