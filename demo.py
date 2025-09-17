import streamlit as st
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import io
import hashlib
import time
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI-Generated Disinfo Detector (Advanced Demo)", layout="wide")
st.title("🤖 AI-Generated Disinfo Detector — Advanced Demo")

def extract_stylometrics(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    words = text.split()
    avg_sent_len = (sum(len(s.split()) for s in sentences) / len(sentences)) if sentences else len(words)
    entropy = -sum((words.count(w)/len(words)) * np.log2(words.count(w)/len(words)+1e-9) for w in set(words)) if words else 0.0
    punct_ratio = sum(1 for c in text if c in "!?.,;:") / max(len(text),1)
    features = {
        "avg_sentence_length": float(avg_sent_len),
        "entropy": float(entropy),
        "punctuation_ratio": float(punct_ratio)
    }
    return features

@st.cache_resource
def generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key

private_key, public_key = generate_rsa_keypair()

def sign_bytes(private_key, data_bytes):
    signature = private_key.sign(
        data_bytes,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return signature

def verify_signature(public_key, data_bytes, signature):
    try:
        public_key.verify(
            signature,
            data_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False

def detect_watermark(text):
    if "[WM]" in text or "watermark:" in text:
        return {"found": True, "method": "embedded_tag", "confidence": 0.98}
    feats = extract_stylometrics(text)
    if feats["entropy"] < 3.2:
        return {"found": True, "method": "heuristic_entropy", "confidence": 0.6}
    # else not found
    return {"found": False, "method": "none", "confidence": 0.05}

def build_sample_graph(flagged_node_indices=None):
    G = nx.erdos_renyi_graph(12, 0.25, seed=42)
    for n in G.nodes():
        score = 0.05 + np.random.rand()*0.3
        if flagged_node_indices and n in flagged_node_indices:
            score = 0.8 + np.random.rand()*0.2
        G.nodes[n]["content_score"] = float(score)
    return G

def graph_features_dataframe(G):
    nodes = list(G.nodes())
    deg = dict(G.degree())
    clustering = nx.clustering(G)
    central = nx.degree_centrality(G)
    features = []
    labels = []
    for n in nodes:
        feats = [
            deg[n],
            clustering[n],
            central[n],
            G.nodes[n].get("content_score", 0.0)
        ]
        label = 1 if (G.nodes[n].get("content_score",0.0) > 0.7 or deg[n] >= 3) else 0
        features.append(feats)
        labels.append(label)
    df = pd.DataFrame(features, columns=["degree","clustering","centrality","content_score"])
    df["label"] = labels
    return df

def simulate_federated_training(num_clients=3, rounds=3):
    """
    Each client trains a tiny logistic regression on its local synthetic stylometric dataset.
    We aggregate model coefficients (weights) by averaging and apply Gaussian noise to simulate DP.
    """
    client_models = []
    client_sizes = []
    rng = np.random.RandomState(0)

    for c in range(num_clients):
        n = 60  
        X = np.column_stack([
            rng.normal(8 + c, 1.5, n),          # avg_sentence_length (vary by client)
            rng.normal(3.0, 0.6, n),            # entropy
            rng.normal(0.2, 0.08, n)            # punct ratio
        ])
        y = ((X[:,0] < 7) & (X[:,1] < 3.2)).astype(int)  
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        client_models.append(model.coef_.ravel().copy())
        client_sizes.append(n)


    sizes = np.array(client_sizes)
    aggregated = np.average(np.vstack(client_models), axis=0, weights=sizes)

    noise_scale = 0.05  
    noise = rng.normal(0, noise_scale, size=aggregated.shape)
    aggregated_dp = aggregated + noise

    return {
        "client_coefs": client_models,
        "aggregated_coef": aggregated,
        "aggregated_coef_dp": aggregated_dp,
        "client_sizes": client_sizes
    }


st.subheader("🔎 Input Section")
user_text = st.text_area("Paste suspicious text:", "Urgent! Your bank account has been suspended. Click this link now to restore access.")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    detect = st.button("🚀 Detect")
with col2:
    upload = st.file_uploader("Upload CSV of posts", type=["csv"])
with col3:
    run_fl = st.button("🔁 Run FL+DP Simulation")


if detect:
    st.subheader("✅ Detection Results")
    feats = extract_stylometrics(user_text)
    text_entropy = feats["entropy"]
    ai_prob = 0.92 if text_entropy < 3.2 else (0.6 if text_entropy < 4.2 else 0.12)
    ai_prob_pct = int(ai_prob * 100)
    wm = detect_watermark(user_text)

    data = {
        "Text Snippet": [user_text],
        "AI Probability": [f"{ai_prob_pct}%"],
        "Risk Level": ["🔴 High" if ai_prob_pct>70 else "🟡 Medium" if ai_prob_pct>40 else "🟢 Low"],
        "Stylometric Notes": [f"entropy={feats['entropy']:.2f}, avg_sent={feats['avg_sentence_length']:.1f}"],
        "Watermark Found": [wm["found"]]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.progress(ai_prob_pct)

    st.subheader("📊 Stylometric Insights")
    st.write(f"- Avg. sentence length: **{feats['avg_sentence_length']:.2f} words**")
    st.write(f"- Entropy: **{feats['entropy']:.2f}**")
    st.write(f"- Punctuation ratio: **{feats['punctuation_ratio']:.2f}**")

    st.subheader("🛡️ Watermark Detection (demo)")
    if wm["found"]:
        st.success(f"Watermark detected by {wm['method']} (confidence {wm['confidence']:.2f})")
    else:
        st.info("No watermark detected.")

    evidence = {
        "id": f"case_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text": user_text,
        "ai_probability": ai_prob,
        "stylometric_features": feats,
        "watermark_detection": wm
    }
    evidence_bytes = json.dumps(evidence, sort_keys=True).encode("utf-8")
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    evidence["hash"] = evidence_hash


    signature = sign_bytes(private_key, evidence_bytes)
    evidence["signature_hex"] = signature.hex()

    st.subheader("📄 Evidence Pack (signed)")
    st.json(evidence)


    evidence_str = json.dumps(evidence, indent=2)
    buf = io.BytesIO(evidence_str.encode())
    st.download_button(label="💾 Download Signed Evidence (JSON)", data=buf, file_name="signed_evidence.json", mime="application/json")


    st.write("---")
    st.subheader("🔐 Signature verification demo")
    ok = verify_signature(public_key, evidence_bytes, signature)
    st.write("Signature valid?" , ok)


    st.subheader("🌐 Graph View & Graph-Classifier (demo)")
    flagged = [0,1,2] if ai_prob>0.7 else [0]
    G = build_sample_graph(flagged_node_indices=flagged)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6,4))
    node_colors = []
    for n in G.nodes():
        node_colors.append("red" if G.nodes[n]["content_score"]>0.7 else "green")
    nx.draw(G, pos=pos, node_color=node_colors, with_labels=True, ax=ax)
    st.pyplot(fig)

    st.write("Training a small RandomForest on graph-derived features (proxy for a GNN).")
    df_feat = graph_features_dataframe(G)
    X = df_feat[["degree","clustering","centrality","content_score"]].values
    y = df_feat["label"].values
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rf.fit(X, y)
    preds = rf.predict(X)
    acc = accuracy_score(y, preds)
    st.write(f"Graph-classifier training accuracy (demo): {acc:.2f}")
    df_feat["pred"] = preds
    st.dataframe(df_feat, use_container_width=True)

if run_fl:
    st.subheader("🔁 Federated Learning + Differential Privacy (simulation)")
    sim = simulate_federated_training(num_clients=3, rounds=3)
    st.write("Client coefficient vectors (per-client):")
    for i,c in enumerate(sim["client_coefs"]):
        st.write(f"Client {i+1} coefs: {np.round(c,3)}")
    st.write("Aggregated coefficients (no DP):", np.round(sim["aggregated_coef"],3))
    st.write("Aggregated coefficients (with DP noise):", np.round(sim["aggregated_coef_dp"],3))
    st.info("Interpretation: DP noise reduces leakage but may slightly reduce utility; in production use calibrated DP and secure aggregation.")

    st.write("---")
    st.write("Toy prediction with aggregated model on a sample stylometric vector:")
    sample = np.array([[7.5, 2.8, 0.18]])  
    score = 1/(1+np.exp(-sample.dot(sim["aggregated_coef"].reshape(-1,1))))
    st.write("Predicted probability (aggregated no-DP):", float(score.ravel()[0]))
    score_dp = 1/(1+np.exp(-sample.dot(sim["aggregated_coef_dp"].reshape(-1,1))))
    st.write("Predicted probability (aggregated with DP):", float(score_dp.ravel()[0]))

st.write("---")
