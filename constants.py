import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
#EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses

# default model
# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"
# GENERATION_CONFIG_NAME = "generation_config.json"

# for HF models
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

# for GPTQ (quantized) models
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
# ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

# for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"

# multi languages
# MODEL_ID = "ai-forever/mGPT"
# MODEL_BASENAME = "pytorch_model.bin"
# GENERATION_CONFIG_NAME = None

# Cannot load this!
# MODEL_ID = "OpenBuddy/openbuddy-llama2-13b-v8.1-fp16"
# MODEL_BASENAME = None
# MODEL_BASENAME = "pytorch_model-00001-of-00003.bin"

# MODEL_ID = "GroNLP/gpt2-small-italian"
MODEL_BASENAME = None
# GENERATION_CONFIG_NAME = "config.json"

# https://github.com/teelinsan/camoscio
MODEL_ID = "decapoda-research/llama-7b-hf"
GENERATION_CONFIG_NAME = None

# Default prompt template
# PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
#     just say that you don't know, don't try to make up an answer.
#
#     {context}
#
#     {history}
#     Question: {question}
#     Helpful Answer:"""

PROMPT_TEMPLATE = """
    Sei un assistente utile ed onesto. Rispondi sempre nel modo più utile possibile utilizzando il testo di contesto fornito. Le tue risposte dovrebbero rispondere alla domanda solo una volta e non avere alcun testo dopo che la risposta è stata data.
    Se una domanda non ha alcun senso o non è coerente, spiega perché invece di rispondere a qualcosa di non corretto. Se non conosci la risposta a una domanda, ti preghiamo di non condividere informazioni false. "

    {context}

    {history}

    Domanda: {question}
    Risposta in italiano:"""


