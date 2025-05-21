# app.py
import streamlit as st
import os
import nltk
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel
from huggingface_hub import login
import torch, os



# --- Download NLTK data once ---
nltk.download("punkt")

# --- Sidebar / Secrets config ---
st.sidebar.markdown("## Model settings")
HUGGINGFACE_TOKEN = st.sidebar.text_input(
    "HF Token", type="password", help="Set your HuggingFace token"
)
BASE_MODEL = st.sidebar.text_input(
    "Base model repo", value="ElijahLiew2/um_p2_fine_tuned_llama"
)

# --- Cached model loader ---
@st.cache_resource
def load_model(base, hf_token, device="cuda" if torch.cuda.is_available() else "cpu"):
    if not hf_token:
        st.error("Provide a HuggingFace token in the sidebar!")
        st.stop()
    login(token=hf_token)


    # login, quant_cfg logic...
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto" if device=="cuda" else None,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        trust_remote_code=True,
        quantization_config=quant_cfg if device=="cuda" else None,
    )
    model = PeftModel.from_pretrained(model, base)
    tokenizer = AutoTokenizer.from_pretrained(base, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=2048,
        repetition_penalty=1.2
    )

    rouge = rouge_scorer.RougeScorer(
        ["rouge1","rouge2","rougeL"], use_stemmer=True
    )
    smooth = SmoothingFunction().method1

    return {
        "model": model,
        "tokenizer": tokenizer,
        "gen_cfg": gen_cfg,
        "rouge": rouge,
        "smooth": smooth,
        "device": device
    }

# --- UI ---
st.title("📄→🤖 QA + ROUGE/BLEU Demo")

# Form to group submit/reset
with st.form("qa_form"):
    uploaded = st.file_uploader("1) Upload .txt context", type=["txt"], key="context_file")
    question = st.text_area("2) Type your question here", key="question")
    submit = st.form_submit_button("Submit")
    reset  = st.form_submit_button("Reset")

# Reset logic
if reset:
    for k in ["context_file","question","answer","rouge_scores","bleu_score"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# On submit
if submit:
    if not uploaded or not question.strip():
        st.error("Please upload a .txt file **and** enter a question.")
    else:
        # read context
        context = uploaded.read().decode("utf-8", errors="ignore")
        # load model
        cfg = load_model(BASE_MODEL, HUGGINGFACE_TOKEN)
        model, tok = cfg["model"], cfg["tokenizer"]
        gen_cfg, rouge_scorer_obj = cfg["gen_cfg"], cfg["rouge"]
        smooth_fn, dev = cfg["smooth"], cfg["device"]

        # build full_prompt        
        only_prompt = """Below is a question about a contract excerpt. \
                    Write a concise answer that satisfies the question.

                    ### Question:
                    {question}

                    ### Contract Excerpt:
                    {context}

                    ### Answer:
                    {answer}"""
                    
        full_prompt = only_prompt.format(
                              question=question, # instruction
                              context=context, # input
                              answer="", # output - leave this blank for generation!
                          )

        inputs = tokenizer(
            [
                full_prompt
            ], return_tensors = "pt").to("cuda")

        with st.spinner("Running inference…"):           
            outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
            answer = tokenizer.batch_decode(outputs)[-1].rsplit("Answer:", 1)[-1].strip()


        # compute metrics (using context as 'reference')
        rouge_scores = rouge_scorer_obj.score(context, answer)
        ref_tokens = nltk.word_tokenize(context.lower())
        cand_tokens = nltk.word_tokenize(answer.lower())
        bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth_fn)

        # cache into session so re-runs keep it
        st.session_state["answer"] = answer
        st.session_state["rouge_scores"] = rouge_scores
        st.session_state["bleu_score"] = bleu

# Display results if present
if "answer" in st.session_state:
    st.subheader("3) Model answer")
    st.write(st.session_state["answer"])

    st.subheader("4) Evaluation")
    r = st.session_state["rouge_scores"]
    st.write(f"- ROUGE-1   P={r['rouge1'].precision:.4f}  R={r['rouge1'].recall:.4f}  F1={r['rouge1'].fmeasure:.4f}")
    st.write(f"- ROUGE-2   F1={r['rouge2'].fmeasure:.4f}")
    st.write(f"- ROUGE-L   F1={r['rougeL'].fmeasure:.4f}")
    st.write(f"- BLEU {st.session_state['bleu_score']:.4f}")

    st.info("Click **Reset** above to start over.")
