# -*- coding: utf-8 -*-
"""Appendix C (requires confidential raw texts; results shipped in results/raw/):
fold-internal vs full-sample text preprocessing.

Re-implemented pipeline (faithful to the documented preprocessing notebook,
without PyKoSpacing): clean Korean text -> Okt POS tokens (Noun/Verb/Adj/Adv)
-> NEG_/EMP_ postprocessing -> stopword removal -> document-frequency filter
(min 5 docs, max 95%) -> TfidfVectorizer(max_features=100, min_df=2,
max_df=0.8, ngram_range=(1,2)).

Two designs, identical models/splits:
  FULL: vocabulary/IDF, frequency filter, and struct standardization fitted
        once on all 6,057 rows (mirrors the paper's design).
  FOLD: all of the above refitted within each training partition only.

Inputs (confidential, NOT distributed): the raw narrative spreadsheet and the
original structured-feature pickle; a row-alignment index maps the pickle row
order to the spreadsheet rows. Outputs: results/raw/foldint_t{5,6,7}.csv.
"""
import re, pickle, sys
import numpy as np, pandas as pd
from collections import Counter
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                             precision_recall_curve, auc)

BASE = './'  # directory holding the confidential raw inputs (not distributed)
OUT = '../results/raw/'

# ---------------- tokenization (once per document) ----------------
STOPWORDS = {
    '은','는','이','가','을','를','에','에서','로','으로','와','과','도','만',
    '부터','까지','같이','처럼','대로','그리고','그래서','하지만','그런데','또한',
    '따라서','그러면','그렇지만','그러나','그럼','그래','근데','그래도','입니다',
    '합니다','해요','있어요','없어요','돼요','돼있어요','있습니다','없습니다',
    '됩니다','있다','없다','되다','하다','이다','아니다','것','수','말','때',
    '곳','쪽','분','명','개','년',
}
NEG = ['안', '못', '아니', '없', '무', '아닌']
EMP = ['매우', '정말', '꼭', '절대', '최대한', '최선', '열심히', '성실히']


def clean_text(t):
    if pd.isna(t):
        return ''
    t = str(t)
    t = re.sub(r'http\S+|www\.\S+', ' ', t)
    t = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', t)
    t = re.sub(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}', ' ', t)
    t = re.sub(r'\d{6}[-\s]?\d{7}', ' ', t)
    t = re.sub(r'\d{3}[-\s]?\d{2}[-\s]?\d{5}', ' ', t)
    t = re.sub(r'[a-zA-Z]+', ' ', t)
    t = re.sub(r'[^가-힣\s]', ' ', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t if len(t) >= 5 else ''


def tokenize(okt, text):
    if not text:
        return []
    toks = [w for w, p in okt.pos(text)
            if p in ('Noun', 'Verb', 'Adjective', 'Adverb')]
    out = []
    for i, tk in enumerate(toks):
        if tk in NEG and i + 1 < len(toks):
            out.append('NEG_' + toks[i + 1]); continue
        if tk in EMP and i + 1 < len(toks):
            out.append('EMP_' + toks[i + 1]); continue
        if i > 0 and (toks[i - 1] in NEG or toks[i - 1] in EMP):
            continue
        out.append(tk)
    return [t for t in out if t not in STOPWORDS]


def build_tokens():
    from konlpy.tag import Okt
    okt = Okt()
    df = pd.read_excel(BASE + 'sentiment_scoring.25.12.30.xlsx')
    mapping = np.load(OUT + 'xlsx_row_mapping.npy')  # pickle row -> raw row
    combined = (df['제목'].map(clean_text) + ' ' +
                df['신청목적'].map(clean_text) + ' ' +
                df['상환계획'].map(clean_text)).str.strip()
    toks = []
    for i, xi in enumerate(mapping):
        toks.append(tokenize(okt, combined.iloc[int(xi)]))
        if (i + 1) % 500 == 0:
            print('tokenized', i + 1, flush=True)
    with open(OUT + 'tokens_pickle_order.pkl', 'wb') as f:
        pickle.dump(toks, f)
    print('tokens saved:', len(toks))


# ---------------- experiment ----------------
def freq_filter_vocab(token_lists, n_docs):
    dfreq = Counter()
    for toks in token_lists:
        for t in set(toks):
            dfreq[t] += 1
    lo, hi = 5, int(n_docs * 0.95)
    return {t for t, c in dfreq.items() if lo <= c <= hi}


def docs_from(tokens, valid):
    return [' '.join(t for t in toks if t in valid) for toks in tokens]


def fit_text(train_docs):
    vec = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8,
                          ngram_range=(1, 2))
    vec.fit(train_docs)
    return vec


def mk():
    return CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                              auto_class_weights='Balanced', random_state=42,
                              verbose=False)


def prauc(yt, pp):
    p, r, _ = precision_recall_curve(yt, pp)
    return auc(r, p)


def run():
    d = pickle.load(open(BASE + 'preprocessed_struct_only_binary.pkl', 'rb'))
    Xr = np.vstack([d['X_train_raw'], d['X_test_raw']])
    y = np.concatenate([np.array(d['y_train']), np.array(d['y_test'])])
    cs = Xr[:, 1]  # credit score, raw scale
    tokens = pickle.load(open(OUT + 'tokens_pickle_order.pkl', 'rb'))
    n = len(y)

    # FULL-design text block: fit everything once on all rows
    vocab_full = freq_filter_vocab(tokens, n)
    docs_full = docs_from(tokens, vocab_full)
    vec_full = fit_text(docs_full)
    T_full = vec_full.transform(docs_full).toarray()
    sc_full = StandardScaler().fit(Xr)
    S_full = sc_full.transform(Xr)

    rows_ov, rows_d2, rows_d3 = [], [], []
    for seed in range(1, 51):
        idx = np.arange(n)
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                                  stratify=y)
        for design in ('FULL', 'FOLD'):
            if design == 'FULL':
                Str, Ste = S_full[tr], S_full[te]
                Ttr, Tte = T_full[tr], T_full[te]
            else:
                sc = StandardScaler().fit(Xr[tr])
                Str, Ste = sc.transform(Xr[tr]), sc.transform(Xr[te])
                toks_tr = [tokens[i] for i in tr]
                vocab = freq_filter_vocab(toks_tr, len(tr))
                vec = fit_text(docs_from(toks_tr, vocab))
                Ttr = vec.transform(docs_from(toks_tr, vocab)).toarray()
                Tte = vec.transform(
                    docs_from([tokens[i] for i in te], vocab)).toarray()
            Mtr = np.hstack([Str, Ttr]); Mte = np.hstack([Ste, Tte])
            m1, m2 = mk(), mk()
            m1.fit(Str, y[tr]); m2.fit(Mtr, y[tr])
            ps = m1.predict_proba(Ste)[:, 1]
            pm = m2.predict_proba(Mte)[:, 1]
            yte = y[te]
            yps = (ps >= 0.5).astype(int); ypm = (pm >= 0.5).astype(int)
            for model, pp, yp in (('S', ps, yps), ('M', pm, ypm)):
                rows_ov.append(dict(seed=seed, design=design, Model=model,
                                    ROC=roc_auc_score(yte, pp),
                                    PR=prauc(yte, pp),
                                    Rec=recall_score(yte, yp),
                                    F1=f1_score(yte, yp)))
            mb = (ps >= 0.3) & (ps <= 0.7)
            for g, mask in (('Marginal', mb), ('Clear', ~mb)):
                for model, pp, yp in (('S', ps, yps), ('M', pm, ypm)):
                    rows_d2.append(dict(seed=seed, design=design, Group=g,
                                        Model=model,
                                        ROC=roc_auc_score(yte[mask], pp[mask]),
                                        Rec=recall_score(yte[mask], yp[mask])))
            cste = cs[te]
            tiers = (('Overall', np.ones(len(yte), bool)),
                     ('High-Risk', cste <= np.percentile(cste, 30)),
                     ('Low-Risk', cste >= np.percentile(cste, 70)))
            for g, mask in tiers:
                fs = int(np.sum((yte[mask] == 1) & (yps[mask] == 0)))
                fm = int(np.sum((yte[mask] == 1) & (ypm[mask] == 0)))
                rows_d3.append(dict(seed=seed, design=design, Group=g,
                                    fs=fs, fm=fm))
        if seed % 5 == 0:
            print('seed', seed, flush=True)
    pd.DataFrame(rows_ov).to_csv(OUT + 'foldint_t5.csv', index=False)
    pd.DataFrame(rows_d2).to_csv(OUT + 'foldint_t6.csv', index=False)
    pd.DataFrame(rows_d3).to_csv(OUT + 'foldint_t7.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    if sys.argv[1:] == ['tokens']:
        build_tokens()
    else:
        run()
