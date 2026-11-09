# Semantic Segmentation and Vision Transformer Fine-Tuning

**Topics:** semantic segmentation, FCN, feature pyramid networks, transfer learning, vision transformers and LoRA.

Two notebook tracks: a baseline EfficientNet-backed FCN and multi-scale FPN segmentation models on PASCAL VOC 2012; plus fine-tuning and parameter-efficient adaptation of a vision transformer for pet image classification.

- `fcn_segmentation.ipynb`: baseline and multi-scale semantic segmentation.
- `vit_finetuning.ipynb`: full/partial fine-tuning and LoRA experiments.
- `a3_dataset_mirrors.py`: dataset download helper from the original project.

**Stack:** Python, PyTorch, torchvision, timm and relevant dataset libraries.

**Reproduction:** Download original datasets and pretrained weights separately. Model checkpoints (hundreds of MB) are omitted. The notebooks retain their original training logic and may require adapting local paths and installing dependencies.
