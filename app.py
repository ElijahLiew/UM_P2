# app.py  (Vertex AI version)
import streamlit as st
import logging
from typing import Dict, Any, List

from google.cloud import aiplatform
from google.oauth2 import service_account
import json
import os



# ── 1) Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



# ── 2) Secrets & config (static across reruns) ──────────────────────────────
# In your Streamlit Cloud repo settings, add under **Secrets**:
#   [secrets]
#   GCP_PROJECT      = "my-project-id"
#   GCP_LOCATION     = "us-central1"       # or asia-southeast1, europe-west4, …
#   VERTEX_ENDPOINT  = "1234567890123456789"       # Endpoint numeric ID
#   SA_KEY           = "*****"             # JSON key **or** leave blank to use workload identity



#PROJECT_ID     = st.secrets.get("GCP_PROJECT")
#LOCATION       = st.secrets.get("GCP_LOCATION", "us-central1")
#ENDPOINT_ID    = st.secrets.get("VERTEX_ENDPOINT")
#SERVICE_ACCOUNT_KEY = st.secrets.get("SA_KEY", None)  # optional

PROJECT_ID     = "custom-history-460319-a4"
LOCATION       = "us-central1"
ENDPOINT_ID    = "2965916123251343360"



#from google.cloud import aiplatform
#ep = aiplatform.Endpoint("2965916123251343360")
#print(f"ep: {ep._gca_endpoint.deployed_models[0].model}")



# 1) Sanity-check the env-var and file
#key_path = ".streamlit/service-account.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
#st.write("Using key file:", key_path)
#st.write("Exists on disk:", os.path.exists(key_path))
#if not os.path.exists(key_path):
#    st.error("Service-account JSON not found! Check your env-var.")
#    st.stop()



# 2) Load the service-account JSON
#with open(key_path, "r") as f:
#    SERVICE_ACCOUNT_KEY = json.load(f)



SERVICE_ACCOUNT_KEY = st.secrets["google"]
logger.info(f"SERVICE_ACCOUNT_KEY: {SERVICE_ACCOUNT_KEY}")



creds = service_account.Credentials.from_service_account_info(
                                                              SERVICE_ACCOUNT_KEY,
                                                              scopes=["https://www.googleapis.com/auth/cloud-platform"],
                                                              )
logger.info(f"creds: {creds}")



aiplatform.init(project=PROJECT_ID, 
                location=LOCATION,
                #credentials=creds,
                )



if not (PROJECT_ID and ENDPOINT_ID):
    st.sidebar.error(
        "🚨 Please add GCP_PROJECT and VERTEX_ENDPOINT to Streamlit secrets."
    )
    st.stop()



# ── 3) Make a singleton Vertex client (cached) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vertex_client(project: str, location: str, sa_key: str | None):
    if sa_key:  # explicit service-account JSON
        logger.info("if sa_key:")
    
        #credentials = service_account.Credentials.from_service_account_info(sa_key)
        
        client = aiplatform.gapic.PredictionServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"},
            credentials=creds,
        )

    endpoint_path = client.endpoint_path(
        project=project, location=location, endpoint=ENDPOINT_ID
    )
    logger.info(f"✅ Vertex client ready - endpoint path: {endpoint_path}")
    return client, endpoint_path



client, endpoint_path = get_vertex_client(PROJECT_ID, LOCATION, SERVICE_ACCOUNT_KEY)



# ── 4) Helper to call your LLM endpoint ─────────────────────────────────────
def call_vertex_llm(
    prompt: str,
    parameters: Dict[str, Any] | None = None,
) -> str:
    """
    Sends a single text prompt to the Vertex endpoint and returns the string
    response.  Adjust `instances` payload or `parameters` to match the way
    you deployed your model (Text-Bison, PaLM 2, custom Llama, etc.).
    """
    if parameters is None:
        # Reasonable generic defaults – tweak for your model
        parameters = {"temperature": 0.2, "maxOutputTokens": 512}

    #instances: List[Dict[str, str]] = [{"content": prompt}]
    instances: List[Dict[str, Any]] = [{
        "messages": [{"author": "user", "content": prompt}]
    }]
    
    logger.info({"instances": instances, "parameters": parameters})

    response = client.predict(
        endpoint=endpoint_path,
        instances=instances,
        parameters=parameters,
    )
    # Each prediction is usually a dict with a 'content' or 'generated_text' key
    pred = response.predictions[0]
    return (
        pred.get("content")
        or pred.get("generated_text")
        or str(pred)  # fallback
    )



# ── 5) UI ───────────────────────────────────────────────────────────────────
st.title("📄→🤖  Vertex AI QA Demo")

uploaded = st.file_uploader("Upload .txt context", type="txt")
question = st.text_area("Enter your question")

if st.button("Submit"):
    if not uploaded or not question.strip():
        st.error("Please upload a file and type a question.")
        st.stop()

    context = uploaded.read().decode("utf-8", "ignore")
    prompt = f"""
Below is a question and context for you to answer.

### Question:
{question}

### Context:
{context}

### Answer:
"""
    with st.spinner("Querying Vertex AI…"):
        answer = call_vertex_llm(prompt)

    # In case the model echoes the prompt, keep only text after 'Answer:' token
    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[-1].strip()

    st.subheader("Answer")
    st.write(answer or "⚠️ No answer returned.")
