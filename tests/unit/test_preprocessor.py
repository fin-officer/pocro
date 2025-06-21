"""
Unit tests for image preprocessor
"""

import os
import tempfile
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from src.core.preprocessor import InvoicePreprocessor


class TestInvoicePreprocessor:
    """Test cases for InvoicePreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return InvoicePreprocessor()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple test image with text-like patterns
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background

        # Add some black rectangles to simulate text areas
        cv2.rectangle(image, (50, 50), (550, 100), (0, 0, 0), -1)  # Header
        cv2.rectangle(image, (50, 150), (300, 200), (0, 0, 0), -1)  # Address
        cv2.rectangle(image, (50, 250), (550, 500), (0, 0, 0), -1)  # Table
        cv2.rectangle(image, (400, 550), (550, 600), (0, 0, 0), -1)  # Total

        # Add some text using OpenCV
        cv2.putText(image, "INVOICE", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return image

    @pytest.fixture
    def noisy_image(self, sample_image):
        """Create a noisy version of the sample image"""
        noise = np.random.randint(0, 50, sample_image.shape, dtype=np.uint8)
        return cv2.add(sample_image, noise)

    @pytest.fixture
    def rotated_image(self, sample_image):
        """Create a rotated version of the sample image"""
        center = (sample_image.shape[1] // 2, sample_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)  # 5 degree rotation
        return cv2.warpAffine(sample_image, rotation_matrix, (sample_image.shape[1], sample_image.shape[0]))

    @pytest.fixture
    def low_contrast_image(self, sample_image):
        """Create a low contrast version of the sample image"""
        return (sample_image * 0.5 + 128).astype(np.uint8)

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor.target_dpi == 300
        assert preprocessor.min_confidence == 0.5
        assert isinstance(preprocessor, InvoicePreprocessor)

    def test_enhance_dpi(self, preprocessor, sample_image):
        """Test DPI enhancement"""
        original_height, original_width = sample_image.shape[:2]

        # Test with default DPI
        enhanced = preprocessor._enhance_dpi(sample_image)

        assert enhanced is not None
        assert enhanced.shape[0] > 0 and enhanced.shape[1] > 0
        assert len(enhanced.shape) == 3  # Should maintain 3 channels

        # Test with specific DPI
        enhanced_300 = preprocessor._enhance_dpi(sample_image, target_dpi=300)
        enhanced_150 = preprocessor._enhance_dpi(sample_image, target_dpi=150)

        # Higher DPI should result in larger image
        assert enhanced_300.shape[0] >= enhanced_150.shape[0]
        assert enhanced_300.shape[1] >= enhanced_150.shape[1]

    def test_remove_noise(self, preprocessor, noisy_image):
        """Test noise removal"""
        denoised = preprocessor._remove_noise(noisy_image)

        assert denoised is not None
        assert denoised.shape == noisy_image.shape

        # Test with grayscale image
        gray_noisy = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        denoised_gray = preprocessor._remove_noise(gray_noisy)

        assert denoised_gray is not None
        assert len(denoised_gray.shape) == 2  # Should remain grayscale

    def test_deskew_image(self, preprocessor, rotated_image):
        """Test image deskewing"""
        deskewed = preprocessor._deskew_image(rotated_image)

        assert deskewed is not None
        assert deskewed.shape[0] > 0 and deskewed.shape[1] > 0

        # Test with grayscale image
        gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        deskewed_gray = preprocessor._deskew_image(gray_rotated)

        assert deskewed_gray is not None

    def test_rotate_image(self, preprocessor, sample_image):
        """Test image rotation"""
        # Test positive rotation
        rotated_pos = preprocessor._rotate_image(sample_image, 10)
        assert rotated_pos is not None
        assert rotated_pos.shape[0] > 0 and rotated_pos.shape[1] > 0

        # Test negative rotation
        rotated_neg = preprocessor._rotate_image(sample_image, -10)
        assert rotated_neg is not None

        # Test zero rotation (should be similar to original)
        rotated_zero = preprocessor._rotate_image(sample_image, 0)
        assert rotated_zero is not None

        # Test with custom background color
        rotated_custom = preprocessor._rotate_image(sample_image, 5, (255, 0, 0))
        assert rotated_custom is not None

    def test_enhance_contrast(self, preprocessor, low_contrast_image):
        """Test contrast enhancement"""
        enhanced = preprocessor._enhance_contrast(low_contrast_image)

        assert enhanced is not None
        assert enhanced.shape == low_contrast_image.shape

        # Check if contrast was actually improved
        original_std = np.std(low_contrast_image)
        enhanced_std = np.std(enhanced)
        assert enhanced_std >= original_std * 0.8  # Allow some tolerance

        # Test with grayscale image
        gray_low_contrast = cv2.cvtColor(low_contrast_image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = preprocessor._enhance_contrast(gray_low_contrast)

        assert enhanced_gray is not None
        assert len(enhanced_gray.shape) == 2

    def test_enhance_contrast_with_custom_params(self, preprocessor, low_contrast_image):
        """Test contrast enhancement with custom parameters"""
        enhanced = preprocessor._enhance_contrast(low_contrast_image, clip_limit=5.0, tile_grid_size=(4, 4))

        assert enhanced is not None
        assert enhanced.shape == low_contrast_image.shape

    def test_normalize_image(self, preprocessor, sample_image):
        """Test image normalization"""
        normalized = preprocessor._normalize_image(sample_image)

        assert normalized is not None
        assert normalized.shape == sample_image.shape
        assert normalized.dtype == np.uint8

        # Check if values are in valid range
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 255

    def test_preprocess_invoice_image_complete_pipeline(self, preprocessor, sample_image):
        """Test complete preprocessing pipeline"""
        processed = preprocessor.preprocess_invoice_image(sample_image)

        assert processed is not None
        assert processed.shape[0] > 0 and processed.shape[1] > 0
        assert len(processed.shape) == 3

        # Ensure the processed image is different from original
        assert not np.array_equal(processed, sample_image)

        # Check that basic properties are maintained
        assert processed.dtype == np.uint8
        assert np.min(processed) >= 0
        assert np.max(processed) <= 255

    def test_preprocess_with_invalid_input(self, preprocessor):
        """Test preprocessing with invalid input"""
        # Test with None
        result = preprocessor.preprocess_invoice_image(None)
        assert result is None

        # Test with empty array
        empty_array = np.array([])
        result = preprocessor.preprocess_invoice_image(empty_array)
        assert np.array_equal(result, empty_array)

        # Test with 1D array
        invalid_array = np.array([1, 2, 3])
        result = preprocessor.preprocess_invoice_image(invalid_array)
        assert np.array_equal(result, invalid_array)

    @pytest.mark.slow
    def test_pdf_to_images_with_mock(self, preprocessor, temp_dir):
        """Test PDF to images conversion with mocked dependencies"""
        # Create a fake PDF file
        fake_pdf_path = temp_dir / "fake.pdf"
        fake_pdf_path.write_bytes(b"fake pdf content")

        # Mock pdf2image.convert_from_path
        with patch("src.core.preprocessor.convert_from_path") as mock_convert:
            mock_image = Image.new("RGB", (800, 600), color="white")
            mock_convert.return_value = [mock_image]

            images = preprocessor.pdf_to_images(str(fake_pdf_path))

            assert len(images) == 1
            assert isinstance(images[0], Image.Image)
            mock_convert.assert_called_once()

    @pytest.mark.slow
    def test_pdf_to_images_fallback_to_pymupdf(self, preprocessor, temp_dir):
        """Test PDF to images conversion with PyMuPDF fallback"""
        fake_pdf_path = temp_dir / "fake.pdf"
        fake_pdf_path.write_bytes(b"fake pdf content")

        # Mock pdf2image to fail and PyMuPDF to succeed
        with patch("src.core.preprocessor.convert_from_path", side_effect=Exception("pdf2image failed")):
            with patch("src.core.preprocessor.fitz") as mock_fitz:
                # Mock PyMuPDF document
                mock_doc = Mock()
                mock_page = Mock()
                mock_pix = Mock()

                mock_doc.__len__.return_value = 1
                mock_doc.load_page.return_value = mock_page
                mock_page.get_pixmap.return_value = mock_pix
                mock_pix.tobytes.return_value = b"fake image data"

                mock_fitz.open.return_value = mock_doc

                # Mock PIL Image.open
                with patch("PIL.Image.open") as mock_image_open:
                    mock_image = Image.new("RGB", (800, 600), color="white")
                    mock_image_open.return_value = mock_image

                    images = preprocessor.pdf_to_images(str(fake_pdf_path))

                    assert len(images) == 1
                    assert isinstance(images[0], Image.Image)

    def test_pdf_to_images_failure(self, preprocessor, temp_dir):
        """Test PDF to images conversion failure"""
        fake_pdf_path = temp_dir / "fake.pdf"
        fake_pdf_path.write_bytes(b"not a real pdf")

        # Mock both pdf2image and PyMuPDF to fail
        with patch("src.core.preprocessor.convert_from_path", side_effect=Exception("pdf2image failed")):
            with patch("src.core.preprocessor.fitz.open", side_effect=Exception("PyMuPDF failed")):
                with pytest.raises(Exception):
                    preprocessor.pdf_to_images(str(fake_pdf_path))

    def test_validate_image_quality_good_image(self, preprocessor, sample_image):
        """Test image quality validation with good image"""
        is_valid = preprocessor.validate_image_quality(sample_image)
        assert is_valid is True

    def test_validate_image_quality_small_image(self, preprocessor):
        """Test image quality validation with small image"""
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        is_valid = preprocessor.validate_image_quality(small_image)
        assert is_valid is False

    def test_validate_image_quality_low_contrast(self, preprocessor):
        """Test image quality validation with low contrast image"""
        # Create uniform gray image (low contrast)
        low_contrast = np.ones((500, 500, 3), dtype=np.uint8) * 128
        is_valid = preprocessor.validate_image_quality(low_contrast)
        assert is_valid is False

    def test_validate_image_quality_blurry_image(self, preprocessor, sample_image):
        """Test image quality validation with blurry image"""
        # Create blurry image
        blurry = cv2.GaussianBlur(sample_image, (51, 51), 20)
        is_valid = preprocessor.validate_image_quality(blurry)
        assert is_valid is False

    def test_validate_image_quality_grayscale(self, preprocessor, sample_image):
        """Test image quality validation with grayscale image"""
        gray_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        is_valid = preprocessor.validate_image_quality(gray_image)
        assert is_valid is True

    def test_validate_image_quality_with_exception(self, preprocessor):
        """Test image quality validation with invalid input"""
        is_valid = preprocessor.validate_image_quality(None)
        assert is_valid is False

        is_valid = preprocessor.validate_image_quality(np.array([]))
        assert is_valid is False

    def test_preprocessing_with_different_input_formats(self, preprocessor):
        """Test preprocessing with different input image formats"""
        # Test with different color spaces
        bgr_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # All should process without errors
        result_bgr = preprocessor.preprocess_invoice_image(bgr_image)
        result_rgb = preprocessor.preprocess_invoice_image(rgb_image)
        result_gray = preprocessor.preprocess_invoice_image(gray_image)

        assert result_bgr is not None
        assert result_rgb is not None
        assert result_gray is not None

        # Results should have correct number of channels
        assert len(result_bgr.shape) == 3
        assert len(result_rgb.shape) == 3
        assert len(result_gray.shape) == 3  # Should be converted to 3-channel

    def test_preprocessing_preserves_aspect_ratio(self, preprocessor):
        """Test that preprocessing preserves aspect ratio"""
        # Create image with specific aspect ratio
        original = np.ones((600, 800, 3), dtype=np.uint8) * 255  # 4:3 ratio
        processed = preprocessor.preprocess_invoice_image(original)

        original_ratio = original.shape[1] / original.shape[0]
        processed_ratio = processed.shape[1] / processed.shape[0]

        # Allow small tolerance for rounding
        assert abs(original_ratio - processed_ratio) < 0.01

    def test_preprocessing_performance(self, preprocessor, sample_image):
        """Test preprocessing performance"""
        import time

        start_time = time.time()
        result = preprocessor.preprocess_invoice_image(sample_image)
        end_time = time.time()

        processing_time = end_time - start_time

        assert result is not None
        assert processing_time < 5.0  # Should complete within 5 seconds

    @pytest.mark.parametrize("dpi", [150, 200, 300, 400])
    def test_enhance_dpi_different_values(self, preprocessor, sample_image, dpi):
        """Test DPI enhancement with different target values"""
        enhanced = preprocessor._enhance_dpi(sample_image, target_dpi=dpi)

        assert enhanced is not None
        assert enhanced.shape[0] > 0 and enhanced.shape[1] > 0

        # Higher DPI should generally result in larger images (unless limited by max size)
        if dpi <= 300:  # Within reasonable range
            area_ratio = (enhanced.shape[0] * enhanced.shape[1]) / (sample_image.shape[0] * sample_image.shape[1])
            expected_ratio = (dpi / 150) ** 2  # Assuming original is ~150 DPI
            assert 0.5 * expected_ratio <= area_ratio <= 2.0 * expected_ratio  # Allow some tolerance

    @pytest.mark.parametrize("angle", [-10, -5, 0, 5, 10, 15])
    def test_rotation_different_angles(self, preprocessor, sample_image, angle):
        """Test rotation with different angles"""
        rotated = preprocessor._rotate_image(sample_image, angle)

        assert rotated is not None
        assert rotated.shape[0] > 0 and rotated.shape[1] > 0

        # For zero rotation, image should be very similar to original
        if angle == 0:
            # Allow for minor differences due to interpolation
            difference = np.mean(np.abs(rotated.astype(float) - sample_image.astype(float)))
            assert difference < 5.0  # Should be very small

    def test_error_handling_in_preprocessing_steps(self, preprocessor):
        """Test error handling in individual preprocessing steps"""
        # Test with problematic input that might cause issues
        problematic_image = np.zeros((10, 10, 3), dtype=np.uint8)  # Very small image

        # All steps should handle this gracefully
        enhanced = preprocessor._enhance_dpi(problematic_image)
        assert enhanced is not None

        denoised = preprocessor._remove_noise(problematic_image)
        assert denoised is not None

        deskewed = preprocessor._deskew_image(problematic_image)
        assert deskewed is not None

        contrast_enhanced = preprocessor._enhance_contrast(problematic_image)
        assert contrast_enhanced is not None

        normalized = preprocessor._normalize_image(problematic_image)
        assert normalized is not None

    def test_memory_efficiency(self, preprocessor):
        """Test that preprocessing doesn't use excessive memory"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process a large image
        large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        result = preprocessor.preprocess_invoice_image(large_image)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        assert result is not None
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500
