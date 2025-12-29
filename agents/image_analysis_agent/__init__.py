from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
from .brain_tumor_agent.brain_tumor_inference import BrainTumorAgent
from .skin_lesion_agent.skin_lesion_inference import SkinLesionClassification

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, 
    and determining their type.
    """
    
    def __init__(self, config):
        # Initialize the 'Gatekeeper' classifier
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        
        # Initialize specialized sub-agents
        self.brain_tumor_agent = BrainTumorAgent(model_path=config.medical_cv.brain_tumor_model_path)
        self.chest_xray_agent = ChestXRayClassification(model_path=config.medical_cv.chest_xray_model_path)
        self.skin_lesion_agent = SkinLesionClassification(model_repo=config.medical_cv.skin_lesion_model_path)
        
        # Path for output if needed by other logic
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
    
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical (e.g., 'chest_xray', 'brain_mri', 'skin_lesion')."""
        return self.image_classifier.classify_image(image_path)
    
    def classify_chest_xray(self, image_path: str) -> str:
        """Calls the Chest X-Ray inference logic."""
        return self.chest_xray_agent.predict(image_path)
    
    def classify_brain_tumor(self, image_path: str) -> str:
        """Calls the Brain Tumor inference logic."""
        return self.brain_tumor_agent.predict(image_path)
    
    def classify_skin_lesion(self, image_path: str):
        """
        Calls the Skin Lesion Swin Transformer model.
        Returns: (label, confidence_score)
        """
        
        return self.skin_lesion_agent.predict(image_path)