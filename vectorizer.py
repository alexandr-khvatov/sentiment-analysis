import numpy as np
from navec import Navec

NAVEC_UNK = '<unk>'
NAVEC_PAD = '<pad>'
navec = Navec.load('saved/embedding/navec_hudlit_v1_12B_500K_300d_100q.tar')
DIMENSION_EMBEDDING = navec.get('<unk>').shape[0]


def vectorize_sentence(sentence: str, vectorizer=navec, unc_token=NAVEC_UNK, pad_token=NAVEC_PAD,
                       max_sequence_length=18) -> np.array:
    tokens = sentence.split()
    embedd_tokens = [vectorizer.get(t, vectorizer[unc_token]) for t in tokens[:max_sequence_length]]
    if len(embedd_tokens) < max_sequence_length:
        embedd_tokens += [vectorizer[pad_token]] * (max_sequence_length - len(embedd_tokens))
    return np.array(embedd_tokens)
