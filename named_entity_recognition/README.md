# Annotation Imputation using NER Training with Adaptive Loss Scaling

This section provides the code to handle the following aspects:
- Loading of the weakly-annotated, synthesized corpus into the `Datasets` corpus format with NER tags (IOB2).
- Training of an NER model with adaptive loss scaling (`Transformers`-based training).
- Applying annotation imputation by re-applying the trained NER model to the weakly-annotated corpus.
- Training of an classical NER model on imputed dataset (`SpaCy`-based training).
- Evaluation on external datasets (partly requires permissions to access to the dataset files, unfortunately).

# Steps to Run the Pipeline
1. Prepare all necessary dependencies within a virtual environment, and other command line tools.
```bash
# Setup env
python3 -m venv env
source env/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Install wget
sudo apt-get install -y wget
```

2. Load and configure the weakly-annotated corpus and the hyperparameters.
```bash
# [1.Option] Use the pre-defined corpus and just adjust the training setups.
./loadDataset.sh
nano setup_cfg.json

# [2.Option] Use your own corpus and setup
cp /path/to/weakly_corpus.jsonl weakly_corpus.jsonl
nano setup_cfg.json
```

3. Run first training for all setups (using adaptive loss scaling).
```bash
# Select the right GPU first (CUDA_VISIBLE_DEVICES=?)
nano setup_run.sh
# Run the training
./setup_run.sh
```
Afterwards, the trained model checkpoints should be in the folder `<SETUP_CONFIG>/results`.

4. Apply the corpus imputation.
```bash
# Select the right GPU first (CUDA_VISIBLE_DEVICES=?)
nano setup_apply.sh
# Run the annotation imputation
./setup_apply.sh
```
Afterwards, the imputed corpus should be at `<SETUP_CONFIG>/dataset_applied.jsonl`.

5. Re-train on the imputed dataset.
```bash
# Select the right GPU first (CUDA_VISIBLE_DEVICES=?)
# You may want to choose another transformer model over 'GerMedBERT/medbert-512' as well.
# IF YOU WANT TO USE MEDBERT-512, MAKE SURE TO AGREE TO THE USAGE AGREEMENT OF THE MODEL AND STORE YOUR API TOKEN TO THE LOCAL HUGGINGFACE CLI
nano setup_retrain.sh
# Run the classical NER training (using SpaCy)
./setup_retrain.sh
```
Afterwards, Spacy's trained NER model (best and last) should be in the folder `<SETUP_CONFIG>/spacy/output/`.

6. [Optional] Run the evaluation on the re-trained models. **THIS WILL FAIL IF YOU DO NOT OWN THE FILES!**
```bash
# Select the right GPU first (CUDA_VISIBLE_DEVICES=?)
nano setup_retrain.sh
# Run the evaluation (also includes external corpus re-applying and inference)
# This also require openjdk-XX-jdk and Maven to be installed, for [BratEval](https://github.com/READ-BioMed/brateval)
./setup_eval.sh
```
