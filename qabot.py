import gradio as gr

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def get_llm():
    model_id = 'mistralai/mistral-medium-2505'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }

    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id = model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id = project_id,
        params = parameters,
    )

    return watsonx_llm

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    
    return loaded_document

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
    )

    chunks = text_splitter.split_documents(data)

    return chunks

def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
    }

    watsonx_embedding = WatsonxEmbeddings(
        model_id = "ibm/slate-125m-english-rtrvr",
        url = "https://us-south.ml.cloud.ibm.com",
        project_id = "skills-network",
        params = embed_params,
    )

    return watsonx_embedding

# def embed_sentence_and_showing_code():
#     code = '''
#         def watsonx_embedding():
#             embed_params = {
#                 EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
#                 EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
#             }

#             watsonx_embedding = WatsonxEmbeddings(
#                 model_id = "ibm/slate-125m-english-rtrvr",
#                 url = "https://us-south.ml.cloud.ibm.com",
#                 project_id = "skills-network",
#                 params = embed_params,
#             )

#             return watsonx_embedding
        
#         embedder = watsonx_embedding()
#         result = embedder.embed_query("hello nice to meet you")
#         print(result[:5])
#     '''

#     embedder = watsonx_embedding()
#     embedding = embedder.embed_query("hello nice to meet you")

#     return code, embedding[:5]

# gr.Interface(
#     fn = embed_sentence_and_showing_code,
#     inputs = None,
#     outputs = ["text", "json"],
#     title = "Code and First 5 Embedding Values"
# ).launch()

def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(documents = chunks, embedding = embedding_model)

    return vectordb

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()

    return retriever

def retriever_qa(file, query):
    llm = get_llm()
    retriever_object = retriever(file)

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever_object,
        return_source_documents = True
    )

    response = qa.invoke({"query": query})

    return response['result']

rag_application = gr.Interface(
    fn = retriever_qa,
    allow_flagging = "never",
    inputs = [
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],

    outputs = gr.Textbox(label = "Answer"),
    title = "PDF Document QA BOT",
    description = "Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."

)

rag_application.launch(server_name = "127.0.0.1", server_port = 7860)