import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),  
            azure_endpoint=os.getenv("azure_endpoint"),  
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.1
        )

class ConversationConfig:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),
            azure_endpoint=os.getenv("azure_endpoint"), 
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.7  
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),  
            azure_endpoint=os.getenv("azure_endpoint"),  
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.3  
        )
        self.context_limit = 15     

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        
        # FIXED: Changed from 1536 to 3072 to match text-embedding-3-large
        self.embedding_dim = 3072  # ← CRITICAL FIX
        
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "/teamspace/studios/this_studio/data/qdrant_db"
        self.doc_local_path = "/teamspace/studios/this_studio/data/docs_db"
        self.parsed_content_dir = "/teamspace/studios/this_studio/data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        
        # Initialize Azure OpenAI Embeddings
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("embedding_deployment_name"),  
            azure_endpoint=os.getenv("embedding_azure_endpoint"),
            api_key=SecretStr(os.getenv("embedding_openai_api_key") or ""),
            api_version=os.getenv("embedding_openai_api_version"),
            dimensions=3072 
        )
        
        # LLM models for RAG operations
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),
            azure_endpoint=os.getenv("azure_endpoint"),
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.3  
        )
        
        self.summarizer_model = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),
            azure_endpoint=os.getenv("azure_endpoint"), 
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.4
        )
        
        self.chunker_model = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),
            azure_endpoint=os.getenv("azure_endpoint"), 
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.0
        )
        
        self.response_generator_model = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),
            azure_endpoint=os.getenv("azure_endpoint"),  
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.3
        )
        
        # Retrieval settings
        self.top_k = 3
        self.vector_search_type = 'similarity'  
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3
        self.max_context_length = 1512
        self.include_sources = True
        self.min_retrieval_confidence = 0.40  
        self.context_limit = 20
        
        # ADDED: Image processing batch size
        self.image_batch_size = 2  # ← Process 2 images at a time to avoid CUDA OOM

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "Yasser18/brain_tumor"
        self.chest_xray_model_path = "Yasser18/chest_x-ray"
        self.skin_lesion_model_path = "Yasser18/swin_skin_lesion"
        self.skin_lesion_segmentation_output_path = "/teamspace/studios/this_studio/data/segmentation_plot.png"
        
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("deployment_name"),
            azure_endpoint=os.getenv("azure_endpoint"), 
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
            api_version=os.getenv("openai_api_version"),
            temperature=0.1
        )

class Config:
    def __init__(self):
        # Initializing sub-configs
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
