# app.py
import streamlit as st
import torch
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ── 1) Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 2) Load HF_TOKEN from Secrets (static across reruns) ────────────────────
# In your Streamlit Cloud repo settings, add under "Secrets":
#   [secrets]
#   HF_TOKEN = "your_huggingface_pat_here"
#HF_TOKEN = st.secrets.get("HF_TOKEN")

HF_TOKEN = "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"

if not HF_TOKEN:
    st.sidebar.error("🚨 Please add your Hugging Face token to Streamlit Secrets as HF_TOKEN")
    st.stop()

# ── 3) Cache the heavy model+tokenizer load once per worker ──────────────────
@st.cache_resource(show_spinner=False)
def get_llm(base_model: str, hf_token: str, device: str):
    logger.info("🔐 Logging in to Hugging Face")
    login(token=hf_token)

    logger.info(f"⏬ Downloading {base_model}")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("✅ Model & tokenizer ready")
    return model, tokenizer

# ── 4) Pick your model & device ─────────────────────────────────────────────
BASE_MODEL = st.sidebar.text_input("HuggingFace model repo", "ElijahLiew2/um_p2_fine_tuned_llama")
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ── 5) This call only happens once! ─────────────────────────────────────────
model, tokenizer = get_llm(BASE_MODEL, HF_TOKEN, device)

# ── 6) The rest of your UI can safely re-run forever ────────────────────────
st.title("📄→🤖 QA + ROUGE/BLEU Demo")

uploaded = st.file_uploader("Upload .txt context", type="txt")
question = st.text_area("Enter your question")
if st.button("Submit"):
    if not uploaded or not question.strip():
        st.error("Please upload a file and type a question.")
    else:
        context = uploaded.read().decode("utf-8", "ignore")
        prompt = f"""
        Below is a question and context for you to answer.
        ### Question:
        {question}
        ### Context:
        {context}
        ### Answer:
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with st.spinner("Generating…"):
            out = model.generate(**inputs, max_new_tokens=512)
        answer = tokenizer.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        st.subheader("Answer")
        st.write(answer)
