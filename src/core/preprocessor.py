"""
Module for preprocessing invoice documents before OCR and data extraction.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

class InvoicePreprocessor:
    """
    Class for preprocessing invoice documents to improve OCR accuracy.
    Handles various preprocessing steps like deskewing, denoising, and contrast adjustment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the InvoicePreprocessor with optional configuration.
        
        Args:
            config: Dictionary containing preprocessing configuration parameters
        """
        self.config = config or {}
        self._setup_default_config()
    
    def _setup_default_config(self) -> None:
        """Set up default configuration parameters if not provided."""
        defaults = {
            'denoise_strength': 10,
            'contrast_factor': 1.2,
            'sharpen_factor': 1.5,
            'threshold_block_size': 11,
            'threshold_constant': 2,
            'deskew': True,
            'remove_noise': True,
            'enhance_contrast': True,
            'convert_to_grayscale': True,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def preprocess(self, image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Apply preprocessing pipeline to the input image.
        
        Args:
            image_path: Path to the input image or numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
        else:
            image = image_path.copy()
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale if needed
        if self.config['convert_to_grayscale'] and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply preprocessing steps
        if self.config['deskew']:
            image = self._deskew(image)
            
        if self.config['remove_noise']:
            image = self._remove_noise(image)
            
        if self.config['enhance_contrast']:
            image = self._enhance_contrast(image)
        
        return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew the image by detecting and correcting rotation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold the image
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        # Calculate minimum area rectangle that contains all non-zero pixels
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle to be between -45 and 45 degrees
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using non-local means denoising.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, 
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            return cv2.fastNlMeansDenoising(
                image,
                h=self.config['denoise_strength'],
                templateWindowSize=7,
                searchWindowSize=21
            )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast and sharpness.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image, 'RGB')
        else:
            pil_image = Image.fromarray(image, 'L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(self.config['contrast_factor'])
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(self.config['sharpen_factor'])
        
        # Convert back to numpy array
        return np.array(enhanced)
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Thresholded image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        return cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY,
            self.config['threshold_block_size'],
            self.config['threshold_constant']
        )

