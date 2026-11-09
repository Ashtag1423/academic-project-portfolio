# Transformers and BERT Prefix Tuning

**Source:** FIT5215 Assignment 2, Transformer notebook (2025).

Implements a Transformer-based text classifier and explores parameter-efficient prefix prompt-tuning with a pretrained BERT model for six-category question classification.

**Reported best experiment (as stated in the original notebook):**

| Metric / setting | Reported value |
|---|---|
| Test accuracy | **94.50%** |
| Prefix length | 10 |
| Learning rate | 0.0002 |
| Batch size | 8 |
| Epochs | 20 |

**File:** `transformer_prefix_tuning.ipynb`

**Note:** The original `best_prefix_tuning_model.pth` checkpoint (~438 MB) has been omitted. The experiment result is reported from the notebook, not independently rerun.
