"""
AI Toolkit versi web (Streamlit): Chatbot, Analisis Dokumen, Machine Learning.

Jalankan lokal : streamlit run app.py
Secrets (opsional tapi disarankan, di Streamlit Cloud: Settings > Secrets):
    ANTHROPIC_API_KEY = "kunci-api-kamu"           # untuk Claude (berbayar)

    # ATAU pakai penyedia gratis yang kompatibel OpenAI (Groq / Gemini / OpenRouter):
    # LLM_PROVIDER = "openai"
    # LLM_API_KEY  = "kunci-gratis-kamu"
    # LLM_BASE_URL = "https://api.groq.com/openai/v1"
    # LLM_MODEL    = "nama-model-dari-penyedia"
    # MAX_DOC_CHARS = 30000                        # model gratis biasanya konteksnya lebih kecil
    APP_PASSWORD      = "password-untuk-masuk"     # kosongkan kalau mau terbuka untuk umum
    MAX_MESSAGES_PER_SESSION = 30                  # batas pesan per pengunjung (kunci server)
"""

import io
import os

import pandas as pd
import streamlit as st

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang cerdas, jujur, dan membantu. "
    "Jawab dalam bahasa yang dipakai pengguna, ringkas tapi lengkap."
)
MAX_ML_ROWS = 50_000

st.set_page_config(page_title="AI Toolkit", page_icon="🤖", layout="wide")


# ----------------------------------------------------------------- config
def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001
        return os.getenv(name, default)


MAX_MSG = int(secret("MAX_MESSAGES_PER_SESSION", 30))
MAX_DOC_CHARS = int(secret("MAX_DOC_CHARS", 150_000))

# Penyedia AI: "claude" (default) atau "openai" (API kompatibel OpenAI: Groq, Gemini, OpenRouter, dll.)
PROVIDER = str(secret("LLM_PROVIDER", "claude")).lower()
LLM_MODEL = secret("LLM_MODEL") or MODEL

# ----------------------------------------------------------------- gerbang password
APP_PASSWORD = secret("APP_PASSWORD")
if APP_PASSWORD and not st.session_state.get("auth"):
    st.title("🔒 AI Toolkit")
    pw = st.text_input("Masukkan password", type="password")
    if pw:
        if pw == APP_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Password salah.")
    st.stop()


# ----------------------------------------------------------------- AI helpers
def get_client():
    key = (st.session_state.get("user_key") or secret("LLM_API_KEY")
           or secret("ANTHROPIC_API_KEY"))
    if not key:
        st.error("Kunci API belum tersedia. Isi di sidebar atau set LLM_API_KEY di Secrets.")
        st.stop()
    if PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=secret("LLM_BASE_URL"))
    import anthropic
    return anthropic.Anthropic(api_key=key)


def check_quota():
    """Batasi pemakaian kalau pakai kunci server (bukan kunci pengunjung)."""
    if st.session_state.get("user_key"):
        return True
    used = st.session_state.get("used", 0)
    if used >= MAX_MSG:
        st.warning(f"Batas {MAX_MSG} pesan per sesi tercapai. "
                   "Masukkan kunci API sendiri di sidebar untuk lanjut.")
        return False
    st.session_state.used = used + 1
    return True


def stream_text(client, messages, system=SYSTEM_PROMPT, max_tokens=2048):
    if PROVIDER == "openai":
        msgs = [{"role": "system", "content": system}] + messages
        stream = client.chat.completions.create(
            model=LLM_MODEL, messages=msgs, max_tokens=max_tokens, stream=True
        )
        for ch in stream:
            if ch.choices and ch.choices[0].delta.content:
                yield ch.choices[0].delta.content
        return
    with client.messages.stream(
        model=LLM_MODEL, max_tokens=max_tokens, system=system, messages=messages
    ) as s:
        for chunk in s.text_stream:
            yield chunk


def ask_ai(client, prompt, max_tokens=1500):
    if PROVIDER == "openai":
        r = client.chat.completions.create(
            model=LLM_MODEL, max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""
    r = client.messages.create(
        model=LLM_MODEL, max_tokens=max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🤖 AI Toolkit")
page = st.sidebar.radio("Menu", ["💬 Chatbot", "📄 Analisis Dokumen", "📊 Machine Learning"])
st.sidebar.divider()
st.sidebar.text_input(
    "Kunci API sendiri (opsional)", type="password", key="user_key",
    help="Kalau diisi, pemakaian memakai kunci kamu dan tanpa batas pesan dari server.",
)
if not st.session_state.get("user_key"):
    st.sidebar.caption(f"Pesan terpakai: {st.session_state.get('used', 0)}/{MAX_MSG}")


# ----------------------------------------------------------------- 1. chatbot
def page_chat():
    st.header("💬 Chatbot")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if st.button("Hapus percakapan"):
        st.session_state.chat = []
        st.rerun()

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Tulis pesan...")
    if prompt and check_quota():
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_text(get_client(), st.session_state.chat))
                st.session_state.chat.append({"role": "assistant", "content": reply})
            except Exception as e:  # noqa: BLE001
                st.session_state.chat.pop()
                st.error(f"Terjadi error: {e}")


# ----------------------------------------------------------------- 2. dokumen
def extract_text(f):
    name, data = f.name.lower(), f.getvalue()
    if name.endswith((".txt", ".md", ".json", ".log")):
        text = data.decode("utf-8", "ignore")
    elif name.endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    elif name.endswith(".docx"):
        import docx
        text = "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    elif name.endswith((".csv", ".xlsx")):
        df = pd.read_csv(io.BytesIO(data)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(data))
        text = (f"Baris: {len(df)}, kolom: {list(df.columns)}\n\n"
                f"{df.describe(include='all').to_string()}\n\n{df.head(20).to_string()}")
    else:
        raise ValueError("Format file tidak didukung.")
    return text[:MAX_DOC_CHARS]


def page_doc():
    st.header("📄 Analisis Dokumen")
    f = st.file_uploader("Unggah dokumen", type=["txt", "md", "pdf", "docx", "csv", "xlsx", "json"])
    if not f:
        return

    if st.session_state.get("doc_name") != f.name:
        try:
            text = extract_text(f)
        except Exception as e:  # noqa: BLE001
            st.error(f"Gagal membaca file: {e}")
            return
        if not text.strip():
            st.error("Dokumen kosong atau berupa hasil scan (teks tidak terbaca).")
            return
        st.session_state.doc_name = f.name
        st.session_state.doc_text = text
        st.session_state.doc_hist = []

    st.caption(f"{f.name} - {len(st.session_state.doc_text):,} karakter terbaca")
    doc_block = f"<dokumen>\n{st.session_state.doc_text}\n</dokumen>"

    if st.button("Buat ringkasan & poin penting") and check_quota():
        prompt = (f"{doc_block}\n\nAnalisis dokumen di atas: (1) ringkasan singkat, "
                  "(2) poin-poin penting, (3) temuan menarik atau hal yang perlu diperhatikan.")
        st.session_state.doc_hist = [{"role": "user", "content": prompt}]
        with st.chat_message("assistant"):
            reply = st.write_stream(stream_text(get_client(), st.session_state.doc_hist, max_tokens=3000))
        st.session_state.doc_hist.append({"role": "assistant", "content": reply})

    st.divider()
    with st.form("tanya_dok", clear_on_submit=True):
        q = st.text_input("Tanya sesuatu tentang dokumen ini")
        sent = st.form_submit_button("Kirim")
    if sent and q and check_quota():
        hist = st.session_state.doc_hist
        if not hist:
            q = f"{doc_block}\n\nPertanyaan: {q}"
        hist.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            reply = st.write_stream(stream_text(get_client(), hist))
        hist.append({"role": "assistant", "content": reply})


# ----------------------------------------------------------------- 3. machine learning
def train_model(df, target):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import classification_report, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = df.dropna(subset=[target])
    y, X = df[target], df.drop(columns=[target])
    is_class = (y.dtype == object or str(y.dtype) in {"bool", "category"}
                or (pd.api.types.is_integer_dtype(y) and y.nunique() <= 10))

    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    if is_class:
        cands = {"Logistic Regression": LogisticRegression(max_iter=1000),
                 "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
                 "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
        scoring = "accuracy"
    else:
        cands = {"Ridge Regression": Ridge(),
                 "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
                 "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
        scoring = "r2"

    strat = y if is_class and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    scores, pipes = {}, {}
    for name, m in cands.items():
        p = Pipeline([("pre", pre), ("model", m)])
        scores[name] = cross_val_score(p, X_tr, y_tr, cv=5, scoring=scoring).mean()
        pipes[name] = p
    best_name = max(scores, key=scores.get)
    best = pipes[best_name].fit(X_tr, y_tr)
    pred = best.predict(X_te)
    report = (classification_report(y_te, pred) if is_class
              else f"R2 = {r2_score(y_te, pred):.4f}\nMAE = {mean_absolute_error(y_te, pred):.4f}")
    return {"pipeline": best, "target": target, "features": list(X.columns),
            "task": "klasifikasi" if is_class else "regresi", "scoring": scoring,
            "scores": scores, "best": best_name, "report": report}


def page_ml():
    st.header("📊 Machine Learning")
    f = st.file_uploader("Unggah data latih (CSV)", type=["csv"])
    if not f:
        return
    df = pd.read_csv(f)
    if len(df) > MAX_ML_ROWS:
        st.warning(f"Data dipotong ke {MAX_ML_ROWS:,} baris pertama.")
        df = df.head(MAX_ML_ROWS)
    st.dataframe(df.head(10), use_container_width=True)

    target = st.selectbox("Kolom target (yang mau diprediksi)", df.columns, index=len(df.columns) - 1)
    if st.button("🚀 Latih model"):
        with st.spinner("Melatih dan membandingkan model..."):
            try:
                st.session_state.bundle = train_model(df, target)
            except Exception as e:  # noqa: BLE001
                st.error(f"Gagal melatih model: {e}")
                return

    b = st.session_state.get("bundle")
    if not b:
        return

    st.success(f"Jenis tugas: {b['task']} | Model terbaik: {b['best']}")
    st.write(f"Perbandingan model (cross-validation, {b['scoring']}):")
    st.bar_chart(pd.Series(b["scores"]))
    st.text(b["report"])

    if st.button("🧠 Jelaskan hasil dengan AI") and check_quota():
        prompt = (f"Saya melatih model {b['task']} untuk memprediksi '{b['target']}'. "
                  f"Skor ({b['scoring']}): { {k: round(v, 4) for k, v in b['scores'].items()} }. "
                  f"Model terbaik: {b['best']}.\nEvaluasi:\n{b['report']}\n\n"
                  "Jelaskan dengan bahasa sederhana: apakah modelnya sudah bagus, apa risikonya, "
                  "dan saran perbaikan.")
        st.write(ask_ai(get_client(), prompt))

    import joblib
    buf = io.BytesIO()
    joblib.dump(b, buf)
    st.download_button("⬇️ Unduh model (.joblib)", buf.getvalue(), "model.joblib")

    st.divider()
    st.subheader("Prediksi data baru")
    st.caption("Memakai model yang baru dilatih di sesi ini. "
               f"Kolom yang dibutuhkan: {', '.join(b['features'])}")
    nf = st.file_uploader("Unggah CSV data baru", type=["csv"], key="newdata")
    if nf:
        new = pd.read_csv(nf)
        missing = [c for c in b["features"] if c not in new.columns]
        if missing:
            st.error(f"Kolom tidak ada di data: {missing}")
        else:
            new[f"prediksi_{b['target']}"] = b["pipeline"].predict(new[b["features"]])
            st.dataframe(new.head(50), use_container_width=True)
            st.download_button("⬇️ Unduh hasil prediksi", new.to_csv(index=False).encode(),
                               "prediksi.csv", "text/csv")


# ----------------------------------------------------------------- router
{"💬 Chatbot": page_chat, "📄 Analisis Dokumen": page_doc, "📊 Machine Learning": page_ml}[page]()
