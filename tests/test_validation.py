"""
Validation and Inference Test Suite for DermScan.
Tests the required verification matrix:
  - Valid skin lesion images (must accept and predict)
  - Blurry images (must reject)
  - Dark images (must reject)
  - Overexposed images (must reject)
  - Documents and screenshots (must reject)
  - Landscapes and non-skin scenery (must reject)
  - Non-skin objects and food (must reject)
"""

import os
import unittest
import torch

from src.predict import predictor
from src.skin_validator import ValidationPipeline


class TestImageValidationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ValidationPipeline(torch.device('cpu'))

    # -----------------------------------------------------------------------
    # Valid Skin Lesion Tests (Expected: ACCEPT)
    # -----------------------------------------------------------------------

    def test_valid_skin_lesion_1(self):
        path = os.path.join('uploads', '1000345938.jpg')
        if os.path.exists(path):
            is_valid, pred, conf, msg = predictor.predict_image(path)
            self.assertTrue(is_valid, f"Expected acceptance for {path}, got rejected: {msg}")
            self.assertIsNotNone(pred, "Expected disease prediction class code")
            self.assertIsNotNone(conf, "Expected model confidence score")
            print(f"[PASS] Valid Skin 1 -> Class: {pred}, Conf: {conf}")

    def test_valid_skin_lesion_2(self):
        path = os.path.join('uploads', 'Superficial_basal_cell_carcinoma.jpg')
        if os.path.exists(path):
            is_valid, pred, conf, msg = predictor.predict_image(path)
            self.assertTrue(is_valid, f"Expected acceptance for {path}, got rejected: {msg}")
            self.assertIsNotNone(pred)
            self.assertIsNotNone(conf)
            print(f"[PASS] Valid Skin 2 -> Class: {pred}, Conf: {conf}")

    def test_valid_skin_lesion_3(self):
        path = os.path.join('uploads', 'Ballarat-Skin-Cancer-Centre-naevus-2.jpg')
        if os.path.exists(path):
            is_valid, pred, conf, msg = predictor.predict_image(path)
            self.assertTrue(is_valid, f"Expected acceptance for {path}, got rejected: {msg}")
            self.assertIsNotNone(pred)
            self.assertIsNotNone(conf)
            print(f"[PASS] Valid Skin 3 -> Class: {pred}, Conf: {conf}")

    # -----------------------------------------------------------------------
    # Negative / Non-Skin Image Tests (Expected: REJECT)
    # -----------------------------------------------------------------------

    def test_reject_extremely_dark_image(self):
        path = os.path.join('tests', 'fixtures', 'dark_image.jpg')
        is_valid, pred, conf, msg = predictor.predict_image(path)
        self.assertFalse(is_valid, "Dark image should be rejected")
        self.assertIsNone(pred, "Rejected images must not display prediction")
        self.assertIsNone(conf, "Rejected images must not display confidence")
        self.assertIn("dark", msg.lower())
        print(f"[PASS] Dark image rejected -> {msg}")

    def test_reject_overexposed_image(self):
        path = os.path.join('tests', 'fixtures', 'overexposed_image.jpg')
        is_valid, pred, conf, msg = predictor.predict_image(path)
        self.assertFalse(is_valid, "Overexposed image should be rejected")
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        self.assertIn("overexposed", msg.lower())
        print(f"[PASS] Overexposed image rejected -> {msg}")

    def test_reject_blurry_image(self):
        path = os.path.join('tests', 'fixtures', 'blurry_skin_image.jpg')
        is_valid, pred, conf, msg = predictor.predict_image(path)
        self.assertFalse(is_valid, "Blurry image should be rejected")
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        self.assertIn("blur", msg.lower())
        print(f"[PASS] Blurry image rejected -> {msg}")

    def test_reject_document_screenshot(self):
        path = os.path.join('tests', 'fixtures', 'document_screenshot.png')
        is_valid, pred, conf, msg = predictor.predict_image(path)
        self.assertFalse(is_valid, "Document screenshot should be rejected")
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        print(f"[PASS] Document/screenshot rejected -> {msg}")

    def test_reject_landscape_scenery(self):
        path = os.path.join('tests', 'fixtures', 'landscape_scenery.jpg')
        is_valid, pred, conf, msg = predictor.predict_image(path)
        self.assertFalse(is_valid, "Landscape scenery should be rejected")
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        print(f"[PASS] Landscape rejected -> {msg}")

    def test_reject_tiny_resolution(self):
        path = os.path.join('tests', 'fixtures', 'tiny_image.jpg')
        is_valid, pred, conf, msg = predictor.predict_image(path)
        self.assertFalse(is_valid, "Tiny image (<64px) should be rejected")
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        print(f"[PASS] Tiny resolution rejected -> {msg}")

    def test_reject_vegetable_food(self):
        path = os.path.join('uploads', '6514330d-ccca-429f-9215-e35211f8bb4d-1_all_6781.jpg')
        if os.path.exists(path):
            is_valid, pred, conf, msg = predictor.predict_image(path)
            self.assertFalse(is_valid, "Food/vegetable image should be rejected")
            self.assertIsNone(pred)
            self.assertIsNone(conf)
            print(f"[PASS] Vegetable/Food rejected -> {msg}")

    def test_reject_menu_document(self):
        path = os.path.join('uploads', 'WhatsApp_Image_2026-08-12_at_1.10.59_AM.jpeg')
        if os.path.exists(path):
            is_valid, pred, conf, msg = predictor.predict_image(path)
            self.assertFalse(is_valid, "Menu/document image should be rejected")
            self.assertIsNone(pred)
            self.assertIsNone(conf)
            print(f"[PASS] Menu/document rejected -> {msg}")


if __name__ == '__main__':
    unittest.main()
