import logging
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinForImageClassification
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ChestXRayClassification:
    def __init__(self, model_path="Yasser18/chest_x-ray", device=None):
        """
        Initialize the Chest X-ray Classification model using Swin Transformer.
        
        Args:
            model_name (str): Hugging Face model ID
            device (torch.device): Device to use for inference
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.device = device if device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and processor from Hugging Face
        self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Model loaded successfully from {model_path}")
    
    def _load_model(self, model_name):
        """Load Swin Transformer model and image processor from Hugging Face Hub."""
        try:
            self.logger.info(f"Loading model from Hugging Face: {model_name}")
            
            # Load image processor
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # Load model
            self.model = SwinForImageClassification.from_pretrained(model_name)
            
            self.logger.info("Model and processor loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model from Hugging Face: {e}")
            raise e
    
    def predict(self, img_path, return_probabilities=False):
        """
        Predict the class of a given chest X-ray image.
        
        Args:
            img_path (str): Path to the image file
            return_probabilities (bool): If True, return probabilities for each class
            
        Returns:
            dict: Contains 'class' prediction and optionally 'probabilities' and 'confidence'
        """
        try:
            # Load and prepare image
            image = Image.open(img_path).convert("RGB")
            self.logger.info(f"Processing image: {img_path}")
            
            # Preprocess image using the processor
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get prediction
                preds = torch.argmax(logits, dim=-1)
                pred_idx = preds.cpu().numpy()[0]
                pred_class = self.class_names[pred_idx]
                
                # Get confidence
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0, pred_idx].cpu().item()
                
                # Prepare result
                result = {
                    'class': pred_class,
                    'confidence': confidence
                }
                
                if return_probabilities:
                    prob_dict = {
                        self.class_names[i]: probs[0, i].cpu().item() 
                        for i in range(len(self.class_names))
                    }
                    result['probabilities'] = prob_dict
                
                self.logger.info(
                    f"Prediction: {pred_class} (Confidence: {confidence:.2%})"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def predict_batch(self, img_paths, return_probabilities=False):
        """
        Predict classes for multiple images.
        
        Args:
            img_paths (list): List of image file paths
            return_probabilities (bool): If True, return probabilities for each class
            
        Returns:
            list: List of prediction results
        """
        results = []
        for img_path in img_paths:
            result = self.predict(img_path, return_probabilities)
            results.append(result)
        return results
    
    def predict_with_visualization(self, img_path, return_probabilities=True):
        """
        Predict and visualize the result on the image.
        
        Args:
            img_path (str): Path to the image file
            return_probabilities (bool): If True, include probability distribution
            
        Returns:
            dict: Prediction result
        """
        result = self.predict(img_path, return_probabilities)
        
        if result is None:
            self.logger.error("Prediction failed")
            return None
        
        # Load and display image
        image = Image.open(img_path).convert("RGB")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.set_title(f"Predicted: {result['class']}\nConfidence: {result['confidence']:.2%}")
        ax1.axis('off')
        
        # Display probabilities if available
        if return_probabilities and 'probabilities' in result:
            probs = result['probabilities']
            classes = list(probs.keys())
            confidences = list(probs.values())
            
            colors = ['#2ecc71' if c == result['class'] else '#e74c3c' for c in classes]
            ax2.bar(classes, confidences, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Probability', fontsize=12)
            ax2.set_title('Class Probabilities', fontsize=12)
            ax2.set_ylim([0, 1])
            ax2.grid(axis='y', alpha=0.3)
            
            # Add percentage labels on bars
            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                ax2.text(i, conf + 0.02, f'{conf:.1%}', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return result


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize classifier with Hugging Face model
    classifier = ChestXRayClassification(model_path="Yasser18/chest_x-ray")
