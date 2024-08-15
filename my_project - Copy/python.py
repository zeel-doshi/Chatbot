import os
import json
import bs4
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
# pages = [
#     "https://www.indianic.com/methodology/project-review/",
#     "https://www.indianic.com/methodology/dedicated-teams/",
#     "https://www.indianic.com/methodology/discovery-process/",
#     "https://www.indianic.com/case-studies/",
#     "https://www.indianic.com/what-we-do/virtual-reality-apps/",
#     "https://www.indianic.com/services/hire-dedicated-developers/",
#     "https://www.indianic.com/what-we-do/quality-assurance-services/",
#     "https://www.indianic.com/what-we-do/ai-and-ml-services/",
#     "https://www.indianic.com/what-we-do/wearable-device-apps-development/",
#     "https://www.indianic.com/what-we-do/internet-of-things/",
#     "https://www.indianic.com/what-we-do/design/",
#     "https://www.indianic.com/what-we-do/web-development/",
#     "https://www.indianic.com/our-work/",
#     "https://www.indianic.com/services/mobile/android-application-development-company/",
#     "https://www.indianic.com/case-studies/",
#     "https://www.indianic.com/services/mobile/ios-app-development/",
#     "https://www.indianic.com/",
#     "https://www.indianic.com/about/",
#     "https://www.indianic.com/focus/",
#     "https://www.indianic.com/industries/",
#     "https://www.indianic.com/what-we-do/",
#     "https://www.indianic.com/methodologies/",
#     "https://www.indianic.com/work/",
#     "https://www.indianic.com/careers/",
#     "https://www.indianic.com/policies/",
#     "https://www.indianic.com/blog/",
#     "https://www.indianic.com/how-to-engage/",
#     "https://www.indianic.com/testimonials/",
#     "https://www.indianic.com/we-work-with/",
#     "https://www.indianic.com/faqs/",
#     "https://www.indianic.com/sitemap/",
#     "https://www.indianic.com/contact/",
#     "https://www.indianic.com/case-studies/cambridge/",
#     "https://www.indianic.com/case-studies/dubai-culture/"
# ]

# # Corresponding classes to parse
# classes = [
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "cbp-spmenu-push",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "cbp-spmenu-push",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "cbp-spmenu-push",
#     "footer",
#     "content insert-remove-container faqs-page-section black-white-theme theme-white",
#     "insert-remove-container body-bg-color black-white-theme about-page-common theme-white",
#     "insert-remove-container body-bg-color black-white-theme about-page-common theme-white",
#     "insert-remove-container black-white-theme theme-white",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "blog-wrap",
#     "content insert-remove-container black-white-theme theme-white about-page-common",
#     "insert-remove-container black-white-theme about-page-common theme-white",
#     "insert-remove-container body-bg-color black-white-theme about-page-common theme-white",
#     "content insert-remove-container faqs-page-section black-white-theme theme-white",
#     "insert-remove-container body-bg-color black-white-theme theme-white",
#     "insert-remove-container black-white-theme theme-white about-page-common",
#     "page-template page-template-template-case-study-brandlogo page-template-template-case-study-brandlogo-php page page-id-42291 page-child parent-pageid-22104 body-theme-white",
#     "page-template page-template-template-case-study-brandlogo page-template-template-case-study-brandlogo-php page page-id-37768 page-child parent-pageid-22104 body-theme-white,content insert-remove-container faqs-page-section black-white-theme theme-white"
# ]

# # Load the web pages
# loader = WebBaseLoader(
#     web_paths=pages,
#     bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=classes))
# )

# # Load the text documents
# text_documents = loader.load()

# text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

# final_documents=text_splitter.split_documents(text_documents)
# print(final_documents[0])


import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# from langchain.schema.document import Document

# all_splits = final_documents
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
#######################
#vectordb = Chroma.from_documents(embedding=hf, persist_directory="C:/Users/Dell/Desktop/my_project/chroma_db", documents=final_documents)
#######################
vectordb = Chroma(embedding_function=hf, persist_directory="C:/Users/Dell/Desktop/my_project/chroma_db")

# Adjust parameters for the retriever
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4,  # Lowered threshold
        "k": 5  # Increased number of retrieved documents
    }
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=518,
    do_sample=False,
    huggingfacehub_api_token='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'
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=False
)

def run_my_rag(qa, query):
    print(f"Query: {query}\n")
    result = qa.run(query)
    print("\nResult: ", result)

### Ask Queries Now
query =""" What is IndiaNIC? """
run_my_rag(qa, query)

query =""" What services are provided by IndiaNIC? """
run_my_rag(qa, query)