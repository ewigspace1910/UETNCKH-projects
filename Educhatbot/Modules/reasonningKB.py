import os
import sys
sys.path.insert(0, os.path.curdir)
from typing import Optional
# import chromadb
# from chromadb.config import Settings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import  LLMChainFilter
import requests

from langchain.llms import OpenAI
import config
import re


def clean_paragraph(paragraph):
    # Replace newline characters with a space
    cleaned_text = paragraph.replace('\n', ' ')
    
    # Remove any unwanted symbols (e.g., multiple spaces, asterisks)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\*+', '', cleaned_text)      # Remove asterisks
    
    # Correct common OCR errors (e.g., 'Heatthe' to 'Heat the')
    # cleaned_text = re.sub(r'Heatthe', 'Heat the', cleaned_text)
    
    # Preserve LaTeX code by skipping any replacements within LaTeX delimiters (e.g., $...$ or \(...\))
    # This is a simple example and may need to be adjusted based on the specific LaTeX patterns in your text
    latex_patterns = re.findall(r'(\$.*?\$|\(.*?\))', cleaned_text)
    for pattern in latex_patterns:
        cleaned_text = cleaned_text.replace(pattern, pattern.replace(' ', ''))

    return cleaned_text



class DocRetrievalKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing 
            all the pdf documents
        """
        self.pdf_source_folder_path = pdf_source_folder_path

    def describe_img(self, imgurl):
        # def summarize_img_by_gpt(base64_image):
        try:
            # Getting the base64 string
            headers = {"Content-Type": "application/json","Authorization": f"Bearer {config.OPENAI_API_KEY_DICT['VISION']}"}

            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Your are acting as an teacher and there is a given image contain information of a question in a examination. \
                            Please extract or describe crucial contents/texts inside this image, If the image is only the illustration for the question please kindly describe it to help the student redraw it. \
                            If the content is not in english. Please dont translate and keep it in the original language. With equations try to convert to latext code. The output shoult be in string format as   \
                            'The given image provide more information that: <image content>'  \
                            "
                    }
                ]
                }
            ],
            "max_tokens": 500
            }

            payload['messages'][0]['content'] += [{                 
                "type": "image_url",
                "image_url": {"url": f"{imgurl}", "detail": "auto"}
                }]
                
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
            output = response['choices'][0]['message']['content']

        except  Exception as e:
            config.LOGGER.error(f"==>[ERR] in resaoningKB.describe_img : {str(e)}")
            config.LOGGER.error(f"==>[ERR] in response : {str(response)}")
            output = ""
        return output

    def load_img(self, base64_image_queue:list):
    # def summarize_img_by_gpt(base64_image):
        try:
            # Getting the base64 string
            headers = {"Content-Type": "application/json","Authorization": f"Bearer {config.OPENAI_API_KEY_DICT['VISION']}"}

            payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Your are acting as a textbook retriever and there are some pages of important documents that you need to store. \
                            Please extract exactly meaning text and numbers (even page number) inside pages as much as posible.  \
                            For existing figures and tables in the page, please extract excaltly text and summarize infomation inside time  \
                            Finally, concatenate sentences then Return finetune document output without any your personal explanations. \
                            If the content is not in english. Please dont translate and keep it in the original language. With equations try to convert to latext code \
                            "
                    }
                ]
                }
            ],
            "max_tokens": 2500
            }
            for base64_image in base64_image_queue:
                payload['messages'][0]['content'] += [{                 
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "auto"}
                    }]
                
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
            output = response['choices'][0]['message']['content']

        except  Exception as e:
            config.LOGGER.error(f"==>[ERR] in resaoningKB.load_img : {str(e)}")
            config.LOGGER.error(f"==>[ERR] in response : {str(response)}")
            output = ""
        return output
    
    def load_txt(self, filepath):
        loader = TextLoader(filepath, encoding="utf8")
        documents = loader.load()
        return documents
    
    def load_pdfs(self, folder_path=None):
        folder_path = self.pdf_source_folder_path if folder_path is None else folder_path
        loader = DirectoryLoader(folder_path)
        loaded_pdfs = loader.load()
        return loaded_pdfs

    def split_documents(self, loaded_docs, **kwargs):
        if config.SPLITTER.upper() == "R":
            splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,chunk_overlap=config.CHUNK_OVERLAP,)
        elif config.SPLITTER.upper() == "C":
            splitter = CharacterTextSplitter(separator=" ", chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        else: raise Exception("[ERR]: config.splitter should be in ['R', 'C']")
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    # def convert_document_to_embeddings(self, chunked_docs, embedder, client_settings=None,
    #                                    persist_directory=config.CHROMA_DB_DIRECTORY):       
    #     vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedder, 
    #                        client_settings=client_settings,)
    #     vector_db.add_documents(chunked_docs)
    #     vector_db.persist()
    #     return vector_db

    # def return_retriever_from_persistant_vector_db(
    #     self, embedder, persist_directory=config.CHROMA_DB_DIRECTORY, client=None, client_settings=None
    # ):
    #     if not os.path.isdir(config.CHROMA_DB_DIRECTORY):
    #         raise NotADirectoryError(
    #             "Please load your vector database first."
    #         )
    #     if client_settings is None:
    #         client_settings = Settings(
    #                 anonymized_telemetry=False,
    #                 allow_reset=True
    #             )
    #     if client is None:
    #         client = chromadb.PersistentClient(path=str(persist_directory), settings=client_settings)
                
    #     vector_db = Chroma(
    #         persist_directory=persist_directory,
    #         embedding_function=embedder,
    #         client=client,
    #         client_settings=client_settings,
    #     )
    #     retriever = vector_db.as_retriever(  search_kwargs={"k": config.k}  )
    #     # return retriever
    #     llm = OpenAI(temperature=0,  model_name=config.model_name)
    #     _filter = LLMChainFilter.from_llm(llm)
    #     # relevant_filter = EmbeddingsFilter(embeddings=embedder, similarity_threshold=0.75)
    #     compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=_filter, base_retriever=retriever
    #     )
    #     return compression_retriever
    
    def return_aws_opensearch_retriever(self, embeddings, indexs:list, use_retriever=True, k=None):
        docsearch = OpenSearchVectorSearch(
            index_name=indexs,
            embedding_function=embeddings,
            opensearch_url=config.OPENSEARCH_URL,
            http_auth=(config.OPENSEARCH_ACCOUNT.split(":")[0], config.OPENSEARCH_ACCOUNT.split(":")[-1]),
        #     use_ssl = False,
        #     verify_certs = False,
        #     ssl_assert_hostname = False,
        #     ssl_show_warn = False,
            timeout = 90
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": config.k if  k is None else k, "score_threshold": 0.15})
        return retriever
        # llm = OpenAI(temperature=0,  model_name=config.model_name)
        # _filter = LLMChainFilter.from_llm(llm)
        # # relevant_filter = EmbeddingsFilter(embeddings=embedder, similarity_threshold=0.75)
        # compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=_filter, base_retriever=retriever
        # )
        # return compression_retriever
    
    def load_KB_into_Opensearch(self, embeddings, docs, index_name):
        docsearch = OpenSearchVectorSearch.from_documents(
            docs,
            embeddings,
            # opensearch_url="host url",
            # http_auth=awsauth,
            timeout=600,
            # use_ssl=True,
            # verify_certs=True,
            # connection_class=RequestsHttpConnection,
            opensearch_url=config.OPENSEARCH_URL,
            http_auth=(config.OPENSEARCH_ACCOUNT.split(":")[0], config.OPENSEARCH_ACCOUNT.split(":")[-1]),
            index_name=(index_name)
        )
        return docsearch
