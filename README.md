# CPR
Abstract:
Multimodal Emotion Recognition in Conversations (ERC) targets the identification of emotions across textual, acoustic, and visual modalities. However, most existing approaches rely on deterministic representations. This limitation is critical, as psychological studies increasingly reveal human emotional expression to be an inherently probabilistic process, sensitive to the statistical distribution of cues...

## Requirements
```
python==3.11.3
torch==2.0.1
torch-geometric==2.3.1
sentence-transformers==2.2.2
```
### Installation
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Sentence Transformer](https://www.sbert.net/)
### Preparing datasets
```
bash run_preprocess.sh
```
### Training
```
bash run_train.sh
```
### Evaluate
```
bash run_eval.sh
```

We thank the authors of MMGCN, CFN-ESA, and DialogueRNN  for sharing their codes.
