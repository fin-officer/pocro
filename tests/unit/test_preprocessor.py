"""
Unit tests for image preprocessor
"""
import numpy as np
import pytest
import cv2

from src.core.preprocessor import InvoicePreprocessor


class TestInvoicePreprocessor:

    @pytest.fixture
    def preprocessor(self):
        return InvoicePreprocessor()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple test image with text-like patterns
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (550, 750), (0, 0, 0), 2)
        cv2.putText(image, "INVOICE", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        return image

    def test_enhance_dpi(self, preprocessor, sample_image):
        """Test DPI enhancement"""
        enhanced = preprocessor._enhance_dpi(sample_image, target_dpi=300)

        assert enhanced is not None
        assert enhanced.shape[0] > 0 and enhanced.shape[1] > 0
        assert len(enhanced.shape) == 3  # Should maintain 3 channels

    def test_remove_noise(self, preprocessor, sample_image):
        """Test noise removal"""
        # Add some noise to the image
        noise = np.random.randint(0, 50, sample_image.shape, dtype=np.uint8)
        noisy_image = cv2.add(sample_image, noise)

        denoised = preprocessor._remove_noise(noisy_image)

        assert denoised is not None
        assert denoised.shape == noisy_image.shape

    def test_deskew_image(self, preprocessor, sample_image):
        """Test image deskewing"""
        # Create a slightly rotated image
        center = (sample_image.shape[1] // 2, sample_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated = cv2.warpAffine(sample_image, rotation_matrix,
                                (sample_image.shape[1], sample_image.shape[0]))

        deskewed = preprocessor._deskew_image(rotated)

        assert deskewed is not None
        assert deskewed.shape == rotated.shape

    def test_enhance_contrast(self, preprocessor, sample_image):
        """Test contrast enhancement"""
        # Create low contrast image
        low_contrast = (sample_image * 0.5 + 128).astype(np.uint8)

        enhanced = preprocessor._enhance_contrast(low_contrast)

        assert enhanced is not None
        assert enhanced.shape == low_contrast.shape

        # Check if contrast was actually improved
        original_std = np.std(low_contrast)
        enhanced_std = np.std(enhanced)
        assert enhanced_std >= original_std

    def test_preprocess_invoice_image(self, preprocessor, sample_image):
        """Test complete preprocessing pipeline"""
        processed = preprocessor.preprocess_invoice_image(sample_image)

        assert processed is not None
        assert processed.shape[0] > 0 and processed.shape[1] > 0
        assert len(processed.shape) == 3

        # Ensure the processed image is different from original
        assert not np.array_equal(processed, sample_image)

    @pytest.mark.asyncio
    async def test_pdf_to_images(self, preprocessor, tmp_path):
        """Test PDF to images conversion"""
        # This test would require a real PDF file
        # For now, we'll just test that the method exists and handles errors

        fake_pdf_path = tmp_path / "fake.pdf"
        fake_pdf_path.write_text("not a real pdf")

        with pytest.raises(Exception):
            preprocessor.pdf_to_images(str(fake_pdf_path))