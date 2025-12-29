import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import logging
import sys

class BrainTumorAgent:
    """
    Agent responsible for Brain Tumor classification using a Vision Transformer (ViT) model.
    Connected to the Hugging Face model repository: Yasser18/brain_tumor
    """
    
    def __init__(self, model_path="Yasser18/brain_tumor", device=None):
        """
        Initialize the agent, load the model and the image processor.
        
        Args:
            model_path (str): The path to the local model or Hugging Face repo ID.
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        # Configure logging to track process flow and errors
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Select hardware device (GPU if available, otherwise CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initializing BrainTumorAgent on device: {self.device}")

        # Load the pre-trained ViT model and its specific image processor
        self.model, self.processor = self._load_model(model_path)
        
        # Move model to the selected device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path):
        """
        Internal method to load the model and processor from Hugging Face or local path.
        """
        try:
            self.logger.info(f"Loading model files from: {model_path}...")
            
            # Auto-fetch model and processor from Hugging Face
            model = ViTForImageClassification.from_pretrained(model_path)
            processor = ViTImageProcessor.from_pretrained(model_path)
            
            self.logger.info("✅ Brain Tumor model and processor loaded successfully.")
            return model, processor
        except Exception as e:
            self.logger.error(f"❌ Failed to load model from {model_path}: {str(e)}")
            raise e

    def predict(self, image_path: str) -> str:
        """
        Analyze an MRI image and predict the type of brain tumor.
        
        Args:
            image_path (str): Path to the image file to be analyzed.
            
        Returns:
            str: The predicted class label (e.g., 'Glioma', 'Meningioma', etc.)
        """
        try:
            # 1. Load and prepare the image
            image = Image.open(image_path).convert('RGB')
            
            # 2. Pre-process the image for the Vision Transformer
            inputs = self.processor(images=image, return_tensors="pt")
            
            # 3. Move input tensors to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 4. Perform Inference (no gradient calculation needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply Softmax to get probabilities for each class
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get the index of the highest probability
                predicted_idx = logits.argmax(-1).item()
            
            # 5. Map index to human-readable label using model configuration
            predicted_class = self.model.config.id2label[predicted_idx]
            confidence = probabilities[0][predicted_idx].item()
            
            self.logger.info(f"Analysis Complete: Predicted {predicted_class} with {confidence:.2%} confidence.")
            
            return predicted_class
            
        except Exception as e:
            self.logger.error(f"An error occurred during image analysis: {str(e)}")
            return "Error: Brain Tumor analysis failed."

# --- Testing Block (Optional) ---
if __name__ == "__main__":
    # Example usage for testing standalone
    agent = BrainTumorAgent(model_path="Yasser18/brain_tumor")
    # result = agent.predict("path_to_your_test_image.jpg")
    # print(f"Final Result: {result}")