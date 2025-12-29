import logging
import torch
from transformers import SwinForImageClassification, AutoImageProcessor
from PIL import Image

class SkinLesionClassification:
    """
    An AI agent that classifies skin lesions using a Swin Transformer model
    hosted on the Hugging Face Model Hub.
    """
    def __init__(self, model_repo="Yasser18/swin_skin_lesion", device=None):
        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Label mapping for the ISIC dataset classes
        self.class_names = ["nv", "mel", "bcc", "akiec", "bkl", "df", "vasc", "scc"]
        
        # Hardware acceleration
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # 1. Load the Image Processor from your HF repository
        self.logger.info(f"Fetching processor from {model_repo}...")
        self.processor = AutoImageProcessor.from_pretrained(model_repo)

        self.logger.info(f"Downloading model weights from {model_repo}...")
        self.model = SwinForImageClassification.from_pretrained(model_repo).to(self.device)

        # Set to evaluation mode
        self.model.eval()
        self.logger.info("Model loaded successfully and ready for inference.")

    def predict(self, img_path):
        """
        Processes an image and returns the predicted class and confidence score.
        """
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert("RGB")
            
            # Preprocess the image for the Swin Transformer
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Get model outputs (logits)
                outputs = self.model(**inputs)
                
                # Convert raw logits to probabilities using Softmax
                probs = torch.softmax(outputs.logits, dim=1)
                
                # Get the index of the highest probability
                idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, idx].item()

            return self.class_names[idx], confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

# --- Standard Execution Block ---
if __name__ == "__main__":
    # Your Hugging Face repository identifier
    REPO_NAME = "Yasser18/swin_skin_lesion"
    
    TEST_IMAGE = "/teamspace/studios/this_studio/data/segmentation_plot.png"
    
    # Initialize the classifier
    classifier = SkinLesionClassification(model_repo=REPO_NAME)
    
    # Perform a test prediction
    try:
        prediction, score = classifier.predict(TEST_IMAGE)
        
        print("\n" + "="*40)
        print(" SKIN LESION ANALYSIS REPORT ")
        print("="*40)
        print(f"MODEL SOURCE : {REPO_NAME}")
        print(f"DIAGNOSIS    : {prediction.upper()}")
        print(f"CONFIDENCE   : {score:.2%}")
        print("="*40)
        
    except FileNotFoundError:
        print(f"Error: The image file was not found at {TEST_IMAGE}")