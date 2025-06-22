"""
Image preprocessing for invoice OCR
"""

import io
import logging
from typing import List, Union

import cv2
import fitz  # PyMuPDF
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


class InvoicePreprocessor:
    """Preprocessor for invoice images and PDFs"""

    def __init__(self):
        self.target_dpi = 300
        self.min_confidence = 0.5

    def preprocess_invoice_image(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for invoice images

        Args:
            image: Input image as numpy array (can be grayscale or color)

        Returns:
            Preprocessed image as 3-channel numpy array
        """
        try:
            # Convert to 3-channel if grayscale
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                if len(image.shape) == 2:
                    # Convert single-channel grayscale to 3-channel
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    # Remove single-channel dimension
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # 1. Enhance DPI
            enhanced = self._enhance_dpi(image, self.target_dpi)

            # 2. Remove noise
            denoised = self._remove_noise(enhanced)

            # 3. Deskew image
            deskewed = self._deskew_image(denoised)

            # 4. Enhance contrast
            contrast_enhanced = self._enhance_contrast(deskewed)

            # 5. Normalize image
            normalized = self._normalize_image(contrast_enhanced)

            # Ensure output is 3-channel
            if len(normalized.shape) == 2:
                normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

            logger.info("Image preprocessing completed successfully")
            return normalized

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image  # Return original if preprocessing fails

    def _enhance_dpi(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """Enhance image resolution to target DPI"""
        height, width = image.shape[:2]

        # Calculate scaling factor based on target DPI
        # Assume original is 150 DPI if not specified
        scale_factor = target_dpi / 150.0

        # Limit maximum size to prevent memory issues
        max_dimension = 4096
        if width * scale_factor > max_dimension or height * scale_factor > max_dimension:
            scale_factor = min(max_dimension / width, max_dimension / height)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Use LANCZOS for high-quality upsampling
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image using Non-local Means Denoising"""
        if len(image.shape) == 3:
            # Color image
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        else:
            # Grayscale image
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 15)

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew using Hough line detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            # Calculate dominant angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.rad2deg(theta) - 90
                angles.append(angle)

            # Use median angle to avoid outliers
            if angles:
                skew_angle = np.median(angles)

                # Only correct if angle is significant
                if abs(skew_angle) > 0.5:
                    return self._rotate_image(image, skew_angle)

        return image

    def _rotate_image(self, image: np.ndarray, angle: float, background_color=(255, 255, 255)) -> np.ndarray:
        """
        Rotate image by given angle

        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees (positive for counter-clockwise)
            background_color: Background color as RGB tuple (default: white)

        Returns:
            Rotated image
        """
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

        # Perform rotation with specified background color
        if len(image.shape) == 2:  # Grayscale image
            border_value = int(np.mean(background_color))
        else:  # Color image
            border_value = background_color

        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=border_value)

    def _enhance_contrast(
        self, image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Args:
            image: Input image as numpy array
            clip_limit: Threshold for contrast limiting (default: 3.0)
            tile_grid_size: Size of grid for histogram equalization (default: 8x8)

        Returns:
            Contrast enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_channel = clahe.apply(l_channel)

            # Merge channels and convert back
            lab = cv2.merge([l_channel, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to enhance readability"""
        # Normalize to 0-255 range
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply slight Gaussian blur to smooth sharp edges
        return cv2.GaussianBlur(normalized, (1, 1), 0)

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF to images

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion

        Returns:
            List of PIL Images
        """
        try:
            # Try pdf2image first (better quality)
            images = convert_from_path(pdf_path, dpi=dpi, fmt="JPEG")
            logger.info(f"Converted PDF to {len(images)} images using pdf2image")
            return images

        except Exception as e:
            logger.warning(f"pdf2image failed: {e}, trying PyMuPDF")

            # Fallback to PyMuPDF
            try:
                logger.debug(f"Opening PDF with PyMuPDF: {pdf_path}")
                doc = fitz.open(pdf_path)
                images = []
                
                # Log detailed document information
                logger.debug(f"PDF type: {type(doc).__name__}")
                logger.debug(f"PDF dir: {dir(doc)}")
                logger.debug(f"PDF has {len(doc)} pages")
                logger.debug(f"PDF page_count: {getattr(doc, 'page_count', 'N/A')}")
                
                # Log document attributes that might affect iteration
                logger.debug(f"PDF has __len__: {hasattr(doc, '__len__')}")
                logger.debug(f"PDF has __getitem__: {hasattr(doc, '__getitem__')}")
                logger.debug(f"PDF has __iter__: {hasattr(doc, '__iter__')}")
                
                try:
                    # Try to get the document's string representation
                    logger.debug(f"PDF string representation: {str(doc)[:200]}...")
                except Exception as e:
                    logger.debug(f"Could not get PDF string representation: {e}")
                
                for page_num in range(len(doc)):
                    try:
                        logger.debug(f"Processing page {page_num}")
                        logger.debug(f"Calling load_page({page_num})...")
                        page = doc.load_page(page_num)
                        logger.debug(f"Got page: {page}")
                        
                        # Create transformation matrix for DPI
                        zoom = dpi / 72.0  # 72 is default DPI
                        mat = fitz.Matrix(zoom, zoom)
                        logger.debug(f"Created transformation matrix: {mat}")
                        
                        # Render page to pixmap
                        logger.debug("Getting pixmap...")
                        pix = page.get_pixmap(matrix=mat)
                        logger.debug(f"Got pixmap: {pix.width}x{pix.height}")
                        
                        # Convert to PIL Image
                        logger.debug("Converting to PPM...")
                        img_data = pix.tobytes("ppm")
                        logger.debug(f"Got PPM data: {len(img_data)} bytes")
                        
                        logger.debug("Creating PIL Image...")
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)
                        logger.debug(f"Added image {len(images)} to results")
                        
                    except Exception as page_error:
                        logger.error(f"Error processing page {page_num}: {page_error}")
                        continue
                
                doc.close()
                logger.info(f"Converted PDF to {len(images)} images using PyMuPDF")
                return images

            except Exception as e2:
                logger.error(f"Both PDF conversion methods failed: {e2}")
                raise Exception(f"Failed to convert PDF: {e2}")

    def validate_image_quality(self, image: np.ndarray) -> bool:
        """
        Validate if image quality is sufficient for OCR

        Args:
            image: Input image

        Returns:
            True if quality is acceptable
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Check image dimensions
            height, width = gray.shape
            if height < 100 or width < 100:
                logger.warning("Image too small for reliable OCR")
                return False

            # Check contrast using standard deviation
            std_dev = np.std(gray)
            if std_dev < 20:
                logger.warning("Image has low contrast")
                return False

            # Check if image is too blurry using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                logger.warning("Image appears to be blurry")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating image quality: {e}")
            return False
