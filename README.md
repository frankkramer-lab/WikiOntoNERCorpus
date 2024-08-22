# Creating Ontology-annotated Corpora from Wikipedia for Medical Named-entity Recognition

This repository contains the files and resources of our paper `Creating Ontology-annotated Corpora from Wikipedia for Medical Named-entity Recognition` for the ACL 2024 BioNLP workshop.

See the paper at: https://aclanthology.org/2024.bionlp-1.47/

## Repository Structure
The repository covers two key subfolders:
- `corpus_synthesis` includes the documentation, code and script examples to:
  * Download, parse, and link Wikipedia and WikiData datasets.
  * Generate the synthesized, weakly-annotated dataset for given SPARQL-defined label classes.
- `named_entity_recognition` includes the code to:
  * Train models with adaptive loss scaling in different setups.
  * Apply Annotation Imputation to obtain a fully-annotated corpus.
  * Train a classical NER model on the fully-annotated corpus.
  * Evaluate the classical, trained model on external datasets.

## Corpus Synthesis
**Note**: If you want to create your own, weakly-annotated corpus, feel free to use our web app at: [https://ontocorpus.misit-augsburg.de/](https://ontocorpus.misit-augsburg.de/)

<kbd><img src="https://github.com/frankkramer-lab/WikiOntoNERCorpus/blob/main/assets/OntoCorpus_Screenshot.png" width="600"></kbd>

## Existing Assets
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

The following SPARQL query was used for the corpus synthesis:
```
# Anything that has an assigned ATC code
SELECT ?item
WHERE
{
?item wdt:P267 ?atccode .
}
```

## Results
Our results from the paper:\
<kbd><img src="https://github.com/frankkramer-lab/WikiOntoNERCorpus/blob/main/assets/results.png" width="600"></kbd>

## Contact
If you have any questions or need additional assets, feel free to open an issue or contact the first author, Johann Frei, via email at: firstname.lastname@informatik.uni-augsburg.de

## Citation
Cite the work with the following BibTex citation:
```
@inproceedings{frei-kramer-2024-creating,
    title = "Creating Ontology-annotated Corpora from {W}ikipedia for Medical Named-entity Recognition",
    author = "Frei, Johann  and
      Kramer, Frank",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Miwa, Makoto  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bionlp-1.47",
    pages = "570--579",
    abstract = "Acquiring annotated corpora for medical NLP is challenging due to legal and privacy constraints and costly annotation efforts, and using annotated public datasets may do not align well to the desired target application in terms of annotation style or language. We investigate the approach of utilizing Wikipedia and WikiData jointly to acquire an unsupervised annotated corpus for named-entity recognition (NER). By controlling the annotation ruleset through WikiData{'}s ontology, we extract custom-defined annotations and dynamically impute weak annotations by an adaptive loss scaling. Our validation on German medication detection datasets yields competitive results. The entire pipeline only relies on open models and data resources, enabling reproducibility and open sharing of models and corpora. All relevant assets are shared on GitHub.",
}
```
