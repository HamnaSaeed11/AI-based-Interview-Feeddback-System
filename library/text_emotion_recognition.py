# ===== Core =====
import numpy as np
import re
import string
import pickle
import traceback

# ===== Scikit-learn =====
from sklearn.base import BaseEstimator, TransformerMixin

# ===== TensorFlow / Keras =====
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    SpatialDropout1D, BatchNormalization, LSTM, Dense
)
from tensorflow.keras import backend as K

# ─────────────────────────────────────────────────────────────────────────────
# Big Five Lexicon
# Based on validated Big Five word markers (Mairesse et al., Pennebaker LIWC,
# John & Srivastava BFI adjective lists).  Each word gets a (+1 or -1) weight.
# ─────────────────────────────────────────────────────────────────────────────

BIG5_LEXICON = {
    # ── EXTRAVERSION (E) ──────────────────────────────────────────────────────
    "extraversion": {
        # positive markers
        "outgoing": 1, "talkative": 1, "sociable": 1, "assertive": 1,
        "energetic": 1, "enthusiastic": 1, "social": 1, "extrovert": 1,
        "extroverted": 1, "bold": 1, "gregarious": 1, "lively": 1,
        "active": 1, "adventurous": 1, "confident": 1, "dominant": 1,
        "spontaneous": 1, "cheerful": 1, "talkative": 1, "chatty": 1,
        "friendly": 1, "leader": 1, "leadership": 1, "exciting": 1,
        "vibrant": 1, "fun": 1, "party": 1, "enjoy": 1, "love": 1,
        "people": 1, "friends": 1, "team": 1, "group": 1, "together": 1,
        "meeting": 1, "gatherings": 1, "presentations": 1, "speaking": 1,
        "network": 1, "networking": 1, "initiative": 1, "outspoken": 1,
        # negative markers
        "introverted": -1, "quiet": -1, "reserved": -1, "shy": -1,
        "withdrawn": -1, "solitary": -1, "isolated": -1, "reclusive": -1,
        "alone": -1, "lonely": -1, "silent": -1, "timid": -1,
    },

    # ── NEUROTICISM (N) ───────────────────────────────────────────────────────
    "neuroticism": {
        # positive markers (high neuroticism)
        "anxious": 1, "nervous": 1, "worried": 1, "fearful": 1,
        "stressed": 1, "tense": 1, "moody": 1, "unstable": 1,
        "emotional": 1, "sensitive": 1, "insecure": 1, "depressed": 1,
        "sad": 1, "angry": 1, "irritable": 1, "frustrated": 1,
        "upset": 1, "distressed": 1, "anxious": 1, "panicked": 1,
        "overwhelmed": 1, "vulnerable": 1, "guilty": 1, "jealous": 1,
        "hostile": 1, "impulsive": 1, "temperamental": 1, "volatile": 1,
        "negative": 1, "pessimistic": 1, "self-conscious": 1,
        "worried": 1, "fear": 1, "doubt": 1, "regret": 1,
        # negative markers (low neuroticism / emotionally stable)
        "calm": -1, "stable": -1, "relaxed": -1, "composed": -1,
        "resilient": -1, "secure": -1, "confident": -1, "optimistic": -1,
        "happy": -1, "content": -1, "balanced": -1, "easygoing": -1,
        "peaceful": -1, "steady": -1, "positive": -1, "cheerful": -1,
    },

    # ── AGREEABLENESS (A) ─────────────────────────────────────────────────────
    "agreeableness": {
        # positive markers
        "cooperative": 1, "helpful": 1, "kind": 1, "empathetic": 1,
        "compassionate": 1, "trusting": 1, "generous": 1, "warm": 1,
        "considerate": 1, "polite": 1, "gentle": 1, "friendly": 1,
        "sympathetic": 1, "supportive": 1, "caring": 1, "nurturing": 1,
        "altruistic": 1, "agreeable": 1, "flexible": 1, "forgiving": 1,
        "patient": 1, "tolerant": 1, "modest": 1, "humble": 1,
        "sincere": 1, "honest": 1, "trustworthy": 1, "loyal": 1,
        "understanding": 1, "sensitive": 1, "collaborative": 1,
        "team": 1, "together": 1, "others": 1, "share": 1, "help": 1,
        # negative markers
        "competitive": -1, "aggressive": -1, "argumentative": -1,
        "stubborn": -1, "selfish": -1, "critical": -1, "suspicious": -1,
        "demanding": -1, "harsh": -1, "cold": -1, "manipulative": -1,
    },

    # ── CONSCIENTIOUSNESS (C) ─────────────────────────────────────────────────
    "conscientiousness": {
        # positive markers
        "organized": 1, "disciplined": 1, "reliable": 1, "responsible": 1,
        "diligent": 1, "hardworking": 1, "ambitious": 1, "efficient": 1,
        "systematic": 1, "thorough": 1, "careful": 1, "precise": 1,
        "punctual": 1, "dependable": 1, "persistent": 1, "dedicated": 1,
        "focused": 1, "goal": 1, "planned": 1, "structured": 1,
        "methodical": 1, "detail": 1, "productive": 1, "achievement": 1,
        "accomplished": 1, "success": 1, "deadline": 1, "schedule": 1,
        "consistent": 1, "committed": 1, "determined": 1, "decisive": 1,
        "prepared": 1, "follow": 1, "complete": 1, "finish": 1,
        # negative markers
        "disorganized": -1, "careless": -1, "lazy": -1, "procrastinate": -1,
        "unreliable": -1, "irresponsible": -1, "forgetful": -1,
        "messy": -1, "impulsive": -1, "scattered": -1, "unfocused": -1,
    },

    # ── OPENNESS (O) ──────────────────────────────────────────────────────────
    "openness": {
        # positive markers
        "creative": 1, "imaginative": 1, "curious": 1, "innovative": 1,
        "artistic": 1, "intellectual": 1, "philosophical": 1,
        "open-minded": 1, "exploring": 1, "learning": 1, "culture": 1,
        "travel": 1, "ideas": 1, "abstract": 1, "theoretical": 1,
        "complex": 1, "diverse": 1, "unconventional": 1, "experimental": 1,
        "novel": 1, "original": 1, "inventive": 1, "visionary": 1,
        "aesthetic": 1, "music": 1, "art": 1, "literature": 1,
        "books": 1, "reading": 1, "writing": 1, "poetry": 1,
        "think": 1, "question": 1, "explore": 1, "discover": 1,
        "knowledge": 1, "insight": 1, "broad": 1, "diverse": 1,
        # negative markers
        "traditional": -1, "conventional": -1, "routine": -1,
        "practical": -1, "concrete": -1, "simple": -1, "narrow": -1,
        "rigid": -1, "conservative": -1, "predictable": -1,
    },
}

TRAIT_ORDER = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]


def lexicon_score(text: str) -> np.ndarray:
    """
    Score text against the Big Five lexicon.
    Returns a numpy array of shape (5,) with values in [0, 1].
    Each score is the normalised sum of lexicon weights found in the text.
    """
    text_lower = text.lower()
    # Remove punctuation for clean word matching
    clean = re.sub(r"[^a-z\s\-]", " ", text_lower)
    words = clean.split()

    raw_scores = []
    for trait in TRAIT_ORDER:
        markers = BIG5_LEXICON[trait]
        score = 0.0
        for word in words:
            if word in markers:
                score += markers[word]
        raw_scores.append(score)

    raw_scores = np.array(raw_scores, dtype=np.float32)

    # Normalise: shift so minimum possible is 0, then scale to [0.05, 0.95]
    # using a sigmoid-like soft normalisation so extreme texts don't clip.
    def soft_norm(x):
        # sigmoid centered at 0, scaled so ±10 hits ~0.73/0.27
        return 1.0 / (1.0 + np.exp(-x * 0.3))

    normed = soft_norm(raw_scores)  # (5,) in (0,1)

    # Renormalise to sum=1 so the dashboard percentages add up cleanly
    total = normed.sum()
    if total > 0:
        normed = normed / total
    else:
        normed = np.ones(5, dtype=np.float32) / 5.0

    return normed


def build_keras_model() -> Model:
    """Rebuild the Keras 2.2.2 architecture for Keras 3 compatibility."""
    inp = Input(shape=(300,), name="input1")
    x = Embedding(input_dim=22347, output_dim=300, name="embedding_15")(inp)
    x = Conv1D(128, 8, activation="relu", padding="same", name="conv1d_38")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="max_pooling1d_38")(x)
    x = SpatialDropout1D(0.3, name="spatial_dropout1d_38")(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, name="batch_normalization_38")(x)
    x = Conv1D(256, 8, activation="relu", padding="same", name="conv1d_39")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="max_pooling1d_39")(x)
    x = SpatialDropout1D(0.3, name="spatial_dropout1d_39")(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, name="batch_normalization_39")(x)
    x = Conv1D(384, 8, activation="relu", padding="same", name="conv1d_40")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="max_pooling1d_40")(x)
    x = SpatialDropout1D(0.3, name="spatial_dropout1d_40")(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, name="batch_normalization_40")(x)
    x = LSTM(180, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, name="lstm_38")(x)
    x = LSTM(180, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, name="lstm_39")(x)
    x = LSTM(180, return_sequences=False, dropout=0.3, recurrent_dropout=0.3, name="lstm_40")(x)
    x = Dense(128, activation="softmax", name="dense_24")(x)
    out = Dense(5, activation="softmax", name="dense_25")(x)
    return Model(inputs=inp, outputs=out)


def keras_model_score(text: str):
    """
    Try to get a prediction from the saved Keras model.
    Returns shape (5,) normalised to sum=1, or None if it fails / has too few tokens.
    The model was trained on movie reviews (IMDB) so it has limited vocab overlap
    with personality text — we only trust it when it matches ≥15 tokens.
    """
    try:
        with open("Models/padding.pickle", "rb") as f:
            tok = pickle.load(f)

        seq = tok.texts_to_sequences([text.lower()])
        n_matched = len(seq[0])
        print(f"[keras_model_score] Matched {n_matched} tokens from tokenizer vocab (num_words={tok.num_words})")

        if n_matched < 15:
            print("[keras_model_score] Too few tokens — skipping Keras model, using lexicon only.")
            return None

        padded = pad_sequences(seq, padding="pre", truncating="pre", maxlen=300)
        model = build_keras_model()
        model.load_weights("Models/Personality_traits_NN.h5")
        pred = model.predict(padded, verbose=0)[0]  # shape (5,)
        K.clear_session()

        total = pred.sum()
        if total > 0:
            pred = pred / total
        return pred.astype(np.float32)

    except Exception as e:
        print(f"[keras_model_score] Error: {e}")
        traceback.print_exc()
        K.clear_session()
        return None


class predict:
    """
    Public API — kept compatible with existing main.py calls.
    predict().run(text, model_name='Personality_traits_NN') returns shape (1, 5).

    Strategy:
      1. Score via validated Big Five lexicon (always works, no vocab limit).
      2. Try Keras model — only trusted when it matches ≥15 tokens.
      3. If Keras succeeds: blend 30% Keras + 70% lexicon.
         If Keras fails or low coverage: 100% lexicon.
    """

    def run(self, text: str, model_name: str = "Personality_traits_NN") -> np.ndarray:
        print(f"[predict.run] Scoring text ({len(text.split())} words)…")

        lex = lexicon_score(text)
        print(f"[predict.run] Lexicon scores: {dict(zip(TRAIT_ORDER, (lex*100).astype(int)))}")

        keras = keras_model_score(text)
        if keras is not None:
            print(f"[predict.run] Keras scores:   {dict(zip(TRAIT_ORDER, (keras*100).astype(int)))}")
            blended = 0.30 * keras + 0.70 * lex
        else:
            blended = lex

        # Renormalise
        blended = blended / blended.sum()
        print(f"[predict.run] Final blended:  {dict(zip(TRAIT_ORDER, (blended*100).astype(int)))}")

        return blended[np.newaxis, :]  # shape (1, 5) — matches old API


# ─────────────────────────────────────────────────────────────────────────────
# Keep NLTKPreprocessor importable (used in main.py get_text_info)
# It now works without NLTK data by falling back to simple tokenisation.
# ─────────────────────────────────────────────────────────────────────────────

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._tokenize(doc) for doc in X])

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        words = text.split()
        # simple stop-word list (no NLTK required)
        stops = {
            "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
            "he","him","his","she","her","hers","it","its","they","them","their",
            "what","which","who","whom","this","that","these","those","am","is","are",
            "was","were","be","been","being","have","has","had","do","does","did",
            "a","an","the","and","but","if","or","because","as","of","at","by",
            "for","with","about","against","between","through","during","to","from",
            "in","out","on","off","over","under","then","once","so","than","too",
            "very","just","also","not","no","nor","only","own","same","than","more",
        }
        return " ".join(w for w in words if w not in stops and len(w) > 2)
