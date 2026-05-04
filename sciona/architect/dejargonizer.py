"""Prompt dejargonizer — normalize ML jargon to canonical vocabulary.

Two modes:
- ``dejargonize_heuristic``: regex-based synonym replacement (fast, no LLM)
- ``dejargonize_llm``: LLM-based rewrite (better, 1 API call)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Synonym table: specific → canonical
# Sorted longest-first at runtime to avoid partial matches.
# ---------------------------------------------------------------------------

_JARGON_MAP: dict[str, str] = {
    # CNN architectures → canonical
    "efficientnetv2": "cnn image backbone",
    "efficientnet": "cnn image backbone",
    "convnext": "cnn image backbone",
    "resnext": "cnn image backbone",
    "resnest": "cnn image backbone",
    "resnet": "cnn image backbone",
    "densenet": "cnn image backbone",
    "mobilenet": "cnn image backbone",
    "mnasnet": "cnn image backbone",
    "nfnet": "cnn image backbone",
    "regnet": "cnn image backbone",
    "inception": "cnn image backbone",
    "vgg": "cnn image backbone",
    "se-resnext": "cnn image backbone with squeeze excitation",
    # Vision transformers
    "swin transformer": "vision transformer backbone",
    "swin-v2": "vision transformer backbone",
    "swin": "vision transformer backbone",
    "vision transformer": "vision transformer backbone",
    "vit": "vision transformer backbone",
    "deit": "vision transformer backbone",
    "beit": "vision transformer backbone",
    "dinov2": "vision transformer backbone",
    "mast3r": "geometric vision foundation model",
    "vggt": "geometric vision foundation model",
    # NLP transformers
    "deberta-v3": "text transformer encoder",
    "deberta": "text transformer encoder",
    "roberta": "text transformer encoder",
    "bert": "text transformer encoder",
    "xlm-roberta": "multilingual text transformer",
    "xlm": "multilingual text transformer",
    "electra": "text transformer encoder",
    "longformer": "long sequence text transformer",
    "bigbird": "long sequence text transformer",
    # Generative models
    "deepseek": "autoregressive language model",
    "llama": "autoregressive language model",
    "mistral": "autoregressive language model",
    "qwen": "autoregressive language model",
    "gemma": "autoregressive language model",
    "gpt-4": "autoregressive language model",
    "gpt": "autoregressive language model",
    # Speech / audio
    "whisper": "speech recognition model",
    "wav2vec": "speech representation model",
    "hubert": "speech representation model",
    # Detectors
    "yolov5": "single stage object detector",
    "yolov8": "single stage object detector",
    "yolox": "single stage object detector",
    "yolo": "single stage object detector",
    "retinanet": "single stage object detector with focal loss",
    "faster r-cnn": "two stage object detector",
    "faster rcnn": "two stage object detector",
    "mask r-cnn": "instance segmentation detector",
    "mask rcnn": "instance segmentation detector",
    "detr": "transformer object detector",
    # Segmentation
    "unet++": "nested encoder decoder segmentation",
    "u-net": "encoder decoder segmentation",
    "unet": "encoder decoder segmentation",
    "deeplabv3+": "encoder decoder segmentation",
    "deeplabv3": "encoder decoder segmentation",
    "fpn": "feature pyramid segmentation",
    "nn-unet": "self-configuring encoder decoder segmentation",
    "nnu-net": "self-configuring encoder decoder segmentation",
    # Boosting
    "lightgbm": "gradient boosting",
    "light gbm": "gradient boosting",
    "xgboost": "gradient boosting",
    "catboost": "gradient boosting",
    "hist gradient boosting": "gradient boosting",
    # Tabular DL
    "tabnet": "tabular deep learning",
    "autogluon": "automl framework",
    "auto-sklearn": "automl framework",
    # Feature matching
    "superpoint": "sparse keypoint detector",
    "superglue": "keypoint matcher",
    "loftr": "dense correspondence matcher",
    "aliked": "sparse keypoint detector",
    "lightglue": "keypoint matcher",
    # Augmentation techniques
    "cutmix": "image augmentation region mixing",
    "mixup": "image augmentation interpolation",
    "mosaic": "image augmentation multi-image stitching",
    "cutout": "image augmentation region masking",
    "gridmask": "image augmentation grid masking",
    "randaugment": "automated image augmentation",
    "autoaugment": "automated image augmentation",
    "specaugment": "spectrogram augmentation masking",
    "albumentations": "image augmentation library",
    # Training techniques
    "test-time augmentation": "test time augmentation averaging",
    "tta": "test time augmentation averaging",
    "stochastic weight averaging": "model weight averaging",
    "swa": "model weight averaging",
    "ohem": "hard example mining",
    "online hard example mining": "hard example mining",
    "knowledge distillation": "teacher student training",
    "pseudo labeling": "self training with confident predictions",
    "pseudo-labeling": "self training with confident predictions",
    "label smoothing": "soft target regularization",
    "multi-sample dropout": "averaged dropout regularization",
    "warmup": "learning rate warmup schedule",
    "cosine annealing": "learning rate cosine schedule",
    "onecyclelr": "learning rate one cycle schedule",
    "one cycle": "learning rate one cycle schedule",
    # Loss functions
    "focal loss": "class imbalance weighted loss",
    "dice loss": "segmentation overlap loss",
    "lovasz": "segmentation submodular loss",
    "lovász": "segmentation submodular loss",
    "arcface": "angular margin metric learning loss",
    "triplet loss": "metric learning contrastive loss",
    "contrastive loss": "metric learning contrastive loss",
    "ctc loss": "connectionist temporal classification loss",
    "ctc": "connectionist temporal classification",
    "crps": "continuous ranked probability score",
    "qwk": "quadratic weighted kappa",
    "binary cross entropy": "binary classification loss",
    "bce": "binary classification loss",
    # Retrieval / similarity
    "faiss": "approximate nearest neighbor index",
    "cosine similarity": "embedding distance metric",
    "annoy": "approximate nearest neighbor index",
    # CV techniques
    "stratifiedkfold": "stratified cross validation",
    "groupkfold": "group cross validation",
    "stratified group k-fold": "stratified group cross validation",
    "5-fold": "five fold cross validation",
    "10-fold": "ten fold cross validation",
    "k-fold": "k fold cross validation",
    # Preprocessing
    "tfidf": "term frequency inverse document frequency",
    "tf-idf": "term frequency inverse document frequency",
    "word2vec": "word embedding",
    "fasttext": "subword embedding",
    "glove": "word embedding",
    "sentence-transformer": "sentence embedding model",
    "sentencetransformer": "sentence embedding model",
    # Formats / tools
    "dicom": "medical image format",
    "nifti": "neuroimaging format",
    "smiles": "molecular string representation",
    "rdkit": "chemistry toolkit",
    "pdb": "protein structure format",
    "openslide": "whole slide image reader",
    "pydicom": "medical image reader",
    "librosa": "audio analysis library",
    "torchaudio": "audio processing library",
    "opencv": "computer vision library",
    # Graph / GNN
    "graph neural network": "graph message passing network",
    "gnn": "graph message passing network",
    "gcn": "graph convolutional network",
    "graphsage": "graph sampling aggregation network",
    "gat": "graph attention network",
    "message passing": "graph message passing",
}

# Pre-sort by length descending so longer matches take priority
_SORTED_JARGON = sorted(_JARGON_MAP.items(), key=lambda x: -len(x[0]))


def dejargonize_heuristic(prompt: str) -> str:
    """Replace known jargon with canonical terms.

    Case-insensitive. Longer patterns matched first to avoid partial hits.
    """
    result = prompt
    for jargon, canonical in _SORTED_JARGON:
        pattern = re.compile(re.escape(jargon), re.IGNORECASE)
        result = pattern.sub(canonical, result)
    return result


async def dejargonize_llm(prompt: str, llm: object) -> str:
    """Use LLM to normalize prompt vocabulary.

    Parameters
    ----------
    prompt : str
        Raw user problem description.
    llm : LLMClient
        Any object with an async ``complete(system, user)`` method.

    Returns
    -------
    str
        Dejargonized prompt in canonical ML vocabulary.
    """
    system = (
        "Rewrite the following machine learning problem description using "
        "generic, canonical terminology. Replace specific model names with "
        "their category (e.g., 'EfficientNet-B4' → 'CNN image backbone', "
        "'LightGBM' → 'gradient boosting'). Replace specific technique "
        "names with their function (e.g., 'CutMix' → 'image augmentation "
        "with region mixing', '5-fold StratifiedGroupKFold' → 'stratified "
        "group cross validation'). Replace framework-specific jargon with "
        "plain descriptions. Keep the problem structure, data description, "
        "metric, and constraints intact. Output only the rewritten text."
    )
    response = await llm.complete(system=system, user=prompt)  # type: ignore[union-attr]
    return response.text  # type: ignore[union-attr]
