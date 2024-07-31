import requests,json,uuid
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from Modules.prompt import prompt
from Modules.chain import Chain
from langchain.document_loaders import TextLoader
from config import TMPPATH as TMPpath
import io
class knowledgebase():
    @staticmethod
    async def parse_pdf(pdf_url):
        """
        this function is used to generate the prompt for the chatbot
        ----------
        parameters:
            context: the context of the chatbot
            question: the question of the chatbot
        return:    
            qa_prompt: the prompt for the chatbot
        """
        response = requests.get(pdf_url)
        with open(os.path.join(TMPpath, "temp.pdf"), "wb") as f:
            f.write(response.content)

        pdf_stream = io.BytesIO(response.content)
        # Open the downloaded PDF file
        with open(os.path.join(TMPpath, "temp.pdf"), "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            chunk_page = []
            for page in pdf_reader.pages:
                text += page.extract_text()
                chunk_page.append(text)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # Remove the temporary file
        # os.remove("temp.pdf")
        kb_id = str(uuid.uuid4())
        temp_text = f"kb-{kb_id}.txt"
        temp_json = f"kb-{kb_id}.json"
        
        with open(os.path.join(TMPpath,temp_text), "w", encoding='utf-8') as f:
            f.write(text)

        with open(os.path.join(TMPpath,temp_json), 'w',encoding='utf-8') as f:
            json.dump(chunks, f,ensure_ascii=False, indent=4)

        return kb_id
    
    async def get_pdf(kb_id):
        """
        this function is used to generate the prompt for the chatbot
        ----------
        parameters:
            context: the context of the chatbot
            question: the question of the chatbot
        return:    
            qa_prompt: the prompt for the chatbot
        """
        try: 
            loader = TextLoader(os.path.join(TMPpath,f"kb-{kb_id}.txt"))
            documents = loader.load()
        except: 
            loader = TextLoader(os.path.join(TMPpath,f"kb-{kb_id}.txt"), encoding="utf8")
            documents = loader.load()
            # loader = TextLoader(text)
            # documents = loader.load()

        
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        return docs
    
    async def generate_question(kb_id):
        """
        this function is used to generate the prompt for the chatbot
        ----------
        parameters:
            context: the context of the chatbot
            question: the question of the chatbot
        return:    
            qa_prompt: the prompt for the chatbot
        """
        # kb_text = f"kb-{kb_id}.txt"
        # with open(f"kb-{kb_id}.txt", 'r',encoding="utf-8") as jsonfile:
        #     chunks = json.load(jsonfile)
        # docs = []
        # with open(f"kb-{kb_id}.json", 'r',encoding="utf-8") as jsonfile:
        #     docs = json.load(jsonfile)

        try: 
            loader = TextLoader(os.path.join(TMPpath,f"kb-{kb_id}.txt"))
            documents = loader.load()
        except: 
            loader = TextLoader(os.path.join(TMPpath,f"kb-{kb_id}.txt"), encoding="utf8")
            documents = loader.load()
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        chain = Chain.create_chain(docs, prompt.generate_qa_prompt())
        
        result = chain({"question": "Summarize the pdf", "chat_history": []})
        return result
    
