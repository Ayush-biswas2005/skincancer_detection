"""
End-to-End Flask Web Application Test Suite.
Verifies form upload, routing, response status codes, and template rendering
for both valid skin specimens and invalid/rejected images.
"""

import os
import unittest
import io
from app import app


class TestDermScanWebApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_homepage_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DermScan Research', response.data)
        self.assertIn(b'Structured skin lesion classification', response.data)

    def test_upload_valid_skin_specimen(self):
        sample_path = os.path.join('uploads', '1000345938.jpg')
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample {sample_path} not available.")

        with open(sample_path, 'rb') as f:
            data = {
                'file': (io.BytesIO(f.read()), '1000345938.jpg')
            }
            response = self.client.post('/predict', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')

        # Must display prediction & confidence
        self.assertIn('Primary Class Code', html)
        self.assertIn('Model Confidence', html)
        self.assertIn('Melanocytic Nevi (nv)', html)
        self.assertIn('Skin Verified', html)
        # Must NOT show invalid image rejection banner
        self.assertNotIn('Validation Gate · Rejected', html)
        print("[PASS] E2E: Valid skin upload accepted and classified accurately.")

    def test_upload_invalid_non_skin_document(self):
        fixture_path = os.path.join('tests', 'fixtures', 'document_screenshot.png')
        with open(fixture_path, 'rb') as f:
            data = {
                'file': (io.BytesIO(f.read()), 'document_screenshot.png')
            }
            response = self.client.post('/predict', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')

        # Must display rejection state
        self.assertIn('Invalid Image', html)
        self.assertIn('Validation Warning', html)
        self.assertIn('Validation Gate · Rejected', html)
        self.assertIn('Upload Valid Skin Image', html)

        # Must NOT display disease predictions or confidence percentages
        self.assertNotIn('Primary Class Code', html)
        self.assertNotIn('Model Confidence', html)
        self.assertNotIn('Probability distribution computed via Softmax', html)
        print("[PASS] E2E: Document screenshot blocked before disease prediction.")

    def test_upload_invalid_dark_image(self):
        fixture_path = os.path.join('tests', 'fixtures', 'dark_image.jpg')
        with open(fixture_path, 'rb') as f:
            data = {
                'file': (io.BytesIO(f.read()), 'dark_image.jpg')
            }
            response = self.client.post('/predict', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')

        self.assertIn('Invalid Image', html)
        self.assertIn('too dark', html.lower())
        self.assertNotIn('Primary Class Code', html)
        self.assertNotIn('Model Confidence', html)
        print("[PASS] E2E: Dark image rejected with quality guidance.")


if __name__ == '__main__':
    unittest.main()
