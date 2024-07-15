# Creating Ontology-annotated Corpora from Wikipedia for Medical Named-entity Recognition

This repository contains the files and resources of our paper `Creating Ontology-annotated Corpora from Wikipedia for Medical Named-entity Recognition` for the ACL 2024 BioNLP workshop.

## Repository Structure
The repository covers two key subfolders:
- `corpus_synthesis` includes the documentation, code and scipt examples to:
  * Download, parse, and link Wikipedia and WikiData datasets.
  * Generate the synthesized, weakly-annotated dataset for given SPARQL-defined label classes.
- `named_entity_recognition` includes the code to:
  * Train models with adaptive loss scaling in different setups.
  * Apply Annotation Imputation to obtain a fully-annotated corpus.
  * Train a classical NER model on the fully-annotated corpus.
  * Evaluate the classical, trained model on external datasets.

## Corpus Synthesis
**Note**: If you want to create your own, weakly-annotated corpus, feel free to use our web app at: [https://ontowiki.misit-augsburg.de/](https://ontowiki.misit-augsburg.de/)

<kbd>![OntoCorpus WebApp Screenshot](assets/OntoCorpus_Screenshot.png | width=250)</kbd>

## Existing Assets:
The ATC corpora from the paper can be found here:

| Label Class | Loss Scaling (unk) | Link                                                                                        |
|-------------|--------------------|---------------------------------------------------------------------------------------------|
| ATC         | raw / not imputed  | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_raw.jsonl)  |
| ATC         | 0.01               | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_0.01.jsonl) |
| ATC         | 0.05               | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_0.05.jsonl) |
| ATC         | 0.1                | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_0.1.jsonl)  |
| ATC         | 0.2                | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_0.2.jsonl)  |
| ATC         | 0.5                | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_0.5.jsonl)  |
| ATC         | 0.8                | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_0.8.jsonl)  |
| ATC         | 1.0                | [Download](https://myweb.rz.uni-augsburg.de/~freijoha/WikiOntoNERCorpus/ATC/ATC_1.0.jsonl)  |

## Contact
If you have any question or need additional assets, feel free to open an issue or contact the first author, Johann Frei, via email at: firstname.lastname@informatik.uni-augsburg.de

## Citation
TBA / to appear / TODO

Frei and Kramer. 2024. Creating Ontology-annotated Corpora from Wikipedia for Medical Named-entity Recognition. In Proceedings of the BIONLP and Shared Tasks Workshop 2024, Bangkok, Thailand. Association for Computational Linguistics.