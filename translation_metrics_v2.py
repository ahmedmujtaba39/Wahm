"""Shared, reproducible metrics for translation quality control."""

from functools import lru_cache

import numpy as np


# Multilingual Sentence-Transformers model with Arabic coverage. Pinning the
# Hugging Face commit makes scores reproducible if the model's main branch moves.
BACKTRANS_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
BACKTRANS_EMBEDDING_REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"


@lru_cache(maxsize=1)
def _sentence_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        BACKTRANS_EMBEDDING_MODEL,
        revision=BACKTRANS_EMBEDDING_REVISION,
    )


def backtrans_cosine_similarities(originals, backtranslations, model=None):
    """Return sentence-embedding cosine similarity for aligned MSA text pairs.

    Empty back-translations produce ``None``. Embeddings are normalized by the
    encoder, so the pairwise dot product is true cosine similarity.
    """
    if len(originals) != len(backtranslations):
        raise ValueError("originals and backtranslations must have equal length")

    scores = [None] * len(originals)
    populated = [i for i, text in enumerate(backtranslations) if str(text).strip()]
    if not populated:
        return scores

    encoder = model or _sentence_encoder()
    left = encoder.encode(
        [originals[i] for i in populated],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    right = encoder.encode(
        [backtranslations[i] for i in populated],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    pairwise_cosines = np.sum(left * right, axis=1)
    for i, similarity in zip(populated, pairwise_cosines):
        scores[i] = float(np.clip(similarity, -1.0, 1.0))
    return scores


def backtrans_cosine_similarity(original, backtranslation, model=None):
    """Return cosine similarity for one original/back-translation pair."""
    return backtrans_cosine_similarities(
        [original], [backtranslation], model=model
    )[0]
