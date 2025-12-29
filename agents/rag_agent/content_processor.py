import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ContentProcessor:
    """
    Processes the parsed content - summarizes images and creates 
    robust text chunks using RecursiveCharacterTextSplitter.
    """
    def __init__(self, config):
        """
        Initialize the content processor.
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model 
        self.chunker_model = config.rag.chunker_model 
        
        # Configuration for batch processing
        self.image_batch_size = getattr(config.rag, 'image_batch_size', 2)
        
        # --- NEW: Initialize the Local Text Splitter ---
        # This replaces the LLM chunking to avoid Azure Content Filter issues
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # Ideal size for medical paragraphs
            chunk_overlap=150,     # Context overlap between chunks
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def summarize_images(self, images: List[str], batch_size: int = None) -> List[str]:
        """
        Summarize images using the provided model with GPU memory management.
        """
        if batch_size is None:
            batch_size = self.image_batch_size
        
        prompt_template = """Describe the image in detail while keeping it concise... 
                           (Same prompt as before)"""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()
        
        results = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(images), batch_size):
            batch = images[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1
            self.logger.info(f"   Processing image batch {current_batch_num}/{total_batches}")
            
            for image in batch:
                try:
                    summary = summary_chain.invoke({"image": image})
                    results.append(summary)
                except Exception as e:
                    self.logger.warning(f"   Error processing image: {str(e)}")
                    results.append("no image summary")
            
            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        return results

    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the document by replacing image placeholders with summaries.
        """
        IMAGE_PLACEHOLDER = ""
        PAGE_BREAK_PLACEHOLDER = ""
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        return self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )

    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(target, f'picture_counter_{counter} {replacement}', 1)
                else:
                    result = result.replace(target, '', 1)
            else:
                break
        return result

    # --- MODIFIED: The New Robust Chunking Method ---
    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document using RecursiveCharacterTextSplitter.
        This avoids sending the full text to the LLM for chunking, 
        preventing Content Filter violations.
        """
        try:
            self.logger.info("   Chunking document using RecursiveCharacterTextSplitter...")
            
            # Perform local semantic-aware splitting
            result_chunks = self.text_splitter.split_text(formatted_document)
            
            self.logger.info(f"   Local chunking successful, created {len(result_chunks)} chunks")
            return result_chunks
            
        except Exception as e:
            self.logger.error(f"   Error during local chunking: {e}")
            # Ultra-fallback: simple string slicing
            return [formatted_document[i:i+1000] for i in range(0, len(formatted_document), 1000)]