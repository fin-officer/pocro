"""
Image processing utilities for invoice OCR
"""

import base64
import io
import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Advanced image processing utilities for invoice OCR"""

    def __init__(self):
        self.default_dpi = 300
        self.min_contrast_threshold = 20
        self.blur_threshold = 100

    def resize_for_ocr(self, image: np.ndarray, target_height: int = 1024) -> np.ndarray:
        """
        Resize image for optimal OCR performance

        Args:
            image: Input image
            target_height: Target height in pixels

        Returns:
            Resized image
        """
        try:
            height, width = image.shape[:2]

            # Calculate aspect ratio
            aspect_ratio = width / height

            # Calculate new dimensions
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

            # Ensure minimum dimensions for OCR
            if new_width < 800:
                new_width = 800
                new_height = int(800 / aspect_ratio)

            # Resize using high-quality interpolation
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            logger.debug(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            return resized

        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return image

    def detect_and_correct_skew(self, image: np.ndarray, max_angle: float = 45.0) -> Tuple[np.ndarray, float]:
        """
        Detect and correct document skew using Hough transform

        Args:
            image: Input image
            max_angle: Maximum angle to consider for correction

        Returns:
            Tuple of (corrected_image, detected_angle)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100, min_theta=0, max_theta=np.pi)

            if lines is None:
                return image, 0.0

            # Calculate angles
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.rad2deg(theta) - 90
                if abs(angle) <= max_angle:
                    angles.append(angle)

            if not angles:
                return image, 0.0

            # Use median angle to avoid outliers
            detected_angle = np.median(angles)

            # Only correct if angle is significant
            if abs(detected_angle) > 0.5:
                corrected = self.rotate_image(image, detected_angle)
                logger.debug(f"Skew corrected: {detected_angle:.2f} degrees")
                return corrected, detected_angle

            return image, detected_angle

        except Exception as e:
            logger.error(f"Failed to correct skew: {e}")
            return image, 0.0

    def rotate_image(
        self, image: np.ndarray, angle: float, background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Rotate image by specified angle

        Args:
            image: Input image
            angle: Rotation angle in degrees
            background_color: Background color for padding

        Returns:
            Rotated image
        """
        try:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            # Calculate rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new bounding dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Perform rotation
            rotated = cv2.warpAffine(
                image,
                M,
                (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=background_color,
            )

            return rotated

        except Exception as e:
            logger.error(f"Failed to rotate image: {e}")
            return image

    def enhance_contrast_adaptive(
        self, image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Enhance image contrast using adaptive histogram equalization

        Args:
            image: Input image
            clip_limit: CLAHE clip limit
            tile_grid_size: Grid size for CLAHE

        Returns:
            Contrast enhanced image
        """
        try:
            if len(image.shape) == 3:
                # Color image - work in LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                l_enhanced = clahe.apply(l_channel)

                # Merge channels
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(image)

            logger.debug("Adaptive contrast enhancement applied")
            return enhanced

        except Exception as e:
            logger.error(f"Failed to enhance contrast: {e}")
            return image

    def remove_noise_advanced(self, image: np.ndarray, strength: str = "medium") -> np.ndarray:
        """
        Advanced noise removal with different strength levels

        Args:
            image: Input image
            strength: Denoising strength ("light", "medium", "strong")

        Returns:
            Denoised image
        """
        try:
            strength_params = {"light": (3, 3, 7, 21), "medium": (10, 10, 7, 21), "strong": (20, 20, 7, 21)}

            h, h_color, template_window, search_window = strength_params.get(strength, strength_params["medium"])

            if len(image.shape) == 3:
                # Color image
                denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h_color, template_window, search_window)
            else:
                # Grayscale image
                denoised = cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)

            logger.debug(f"Noise removal applied with {strength} strength")
            return denoised

        except Exception as e:
            logger.error(f"Failed to remove noise: {e}")
            return image

    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in image using MSER

        Args:
            image: Input image

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Create MSER detector
            mser = cv2.MSER_create(
                _delta=5,
                _min_area=60,
                _max_area=14400,
                _max_variation=0.25,
                _min_diversity=0.2,
                _max_evolution=200,
                _area_threshold=1.01,
                _min_margin=0.003,
                _edge_blur_size=5,
            )

            # Detect regions
            regions, _ = mser.detectRegions(gray)

            # Convert to bounding boxes
            boxes = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                boxes.append((x, y, w, h))

            logger.debug(f"Detected {len(boxes)} text regions")
            return boxes

        except Exception as e:
            logger.error(f"Failed to detect text regions: {e}")
            return []

    def crop_to_content(self, image: np.ndarray, padding: int = 20) -> np.ndarray:
        """
        Crop image to content area, removing excess white space

        Args:
            image: Input image
            padding: Padding to add around content

        Returns:
            Cropped image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Threshold to binary
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image

            # Get bounding box of all content
            x_min, y_min = float("inf"), float("inf")
            x_max, y_max = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            # Add padding
            height, width = image.shape[:2]
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)

            # Crop image
            cropped = image[y_min:y_max, x_min:x_max]

            logger.debug(f"Image cropped from {width}x{height} to {x_max-x_min}x{y_max-y_min}")
            return cropped

        except Exception as e:
            logger.error(f"Failed to crop to content: {e}")
            return image

    def apply_morphological_operations(self, image: np.ndarray, operation: str = "close") -> np.ndarray:
        """
        Apply morphological operations to clean up text

        Args:
            image: Input binary image
            operation: Operation type ("open", "close", "tophat", "blackhat")

        Returns:
            Processed image
        """
        try:
            # Define kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            operations = {
                "open": cv2.MORPH_OPEN,
                "close": cv2.MORPH_CLOSE,
                "tophat": cv2.MORPH_TOPHAT,
                "blackhat": cv2.MORPH_BLACKHAT,
            }

            op_type = operations.get(operation, cv2.MORPH_CLOSE)
            processed = cv2.morphologyEx(image, op_type, kernel)

            logger.debug(f"Morphological {operation} operation applied")
            return processed

        except Exception as e:
            logger.error(f"Failed to apply morphological operation: {e}")
            return image

    def convert_to_binary(self, image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Convert image to binary using various methods

        Args:
            image: Input image
            method: Binarization method ("otsu", "adaptive", "mean")

        Returns:
            Binary image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            elif method == "adaptive":
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            elif method == "mean":
                mean_val = np.mean(gray)
                _, binary = cv2.threshold(gray, mean_val, 255, cv2.THRESH_BINARY)

            else:
                # Default to Otsu
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            logger.debug(f"Binary conversion using {method} method")
            return binary

        except Exception as e:
            logger.error(f"Failed to convert to binary: {e}")
            return image

    def calculate_image_quality_metrics(self, image: np.ndarray) -> dict:
        """
        Calculate various image quality metrics

        Args:
            image: Input image

        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Calculate metrics
            metrics = {}

            # Sharpness (Laplacian variance)
            metrics["sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Contrast (standard deviation)
            metrics["contrast"] = np.std(gray)

            # Brightness (mean intensity)
            metrics["brightness"] = np.mean(gray)

            # Signal-to-noise ratio estimation
            mean = np.mean(gray)
            std = np.std(gray)
            metrics["snr"] = mean / std if std > 0 else 0

            # Dynamic range
            metrics["dynamic_range"] = np.max(gray) - np.min(gray)

            # Estimate overall quality score (0-1)
            sharpness_score = min(metrics["sharpness"] / 1000, 1.0)  # Normalize
            contrast_score = min(metrics["contrast"] / 128, 1.0)
            brightness_score = 1.0 - abs(metrics["brightness"] - 128) / 128  # Optimal around 128

            metrics["overall_quality"] = (sharpness_score + contrast_score + brightness_score) / 3

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
            return {}

    def image_to_base64(self, image: np.ndarray, format: str = "JPEG") -> str:
        """
        Convert image to base64 string

        Args:
            image: Input image
            format: Output format (JPEG, PNG)

        Returns:
            Base64 encoded string
        """
        try:
            # Convert BGR to RGB for PIL
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image)

            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            buffer.seek(0)

            base64_string = base64.b64encode(buffer.getvalue()).decode()
            return base64_string

        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            return ""

    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """
        Convert base64 string to image

        Args:
            base64_string: Base64 encoded image

        Returns:
            Image as numpy array
        """
        try:
            # Decode base64
            image_data = base64.b64decode(base64_string)

            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))

            # Convert to numpy array
            image_array = np.array(pil_image)

            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                return image_bgr

            return image_array

        except Exception as e:
            logger.error(f"Failed to convert base64 to image: {e}")
            return np.array([])


# Global image processor instance
image_processor = ImageProcessor()


def preprocess_for_ocr(image: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """
    Complete preprocessing pipeline for OCR

    Args:
        image: Input image
        aggressive: Whether to apply aggressive processing

    Returns:
        Preprocessed image
    """
    try:
        # Step 1: Resize for optimal OCR
        processed = image_processor.resize_for_ocr(image)

        # Step 2: Skew correction
        processed, _ = image_processor.detect_and_correct_skew(processed)

        # Step 3: Noise removal
        strength = "strong" if aggressive else "medium"
        processed = image_processor.remove_noise_advanced(processed, strength)

        # Step 4: Contrast enhancement
        processed = image_processor.enhance_contrast_adaptive(processed)

        # Step 5: Crop to content (optional for aggressive mode)
        if aggressive:
            processed = image_processor.crop_to_content(processed)

        return processed

    except Exception as e:
        logger.error(f"OCR preprocessing failed: {e}")
        return image


def assess_image_quality(image: np.ndarray) -> Tuple[bool, dict]:
    """
    Assess if image quality is suitable for OCR

    Args:
        image: Input image

    Returns:
        Tuple of (is_suitable, quality_metrics)
    """
    try:
        metrics = image_processor.calculate_image_quality_metrics(image)

        # Quality thresholds
        min_sharpness = 100
        min_contrast = 20
        min_quality_score = 0.3

        is_suitable = (
            metrics.get("sharpness", 0) >= min_sharpness
            and metrics.get("contrast", 0) >= min_contrast
            and metrics.get("overall_quality", 0) >= min_quality_score
        )

        return is_suitable, metrics

    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return False, {}
