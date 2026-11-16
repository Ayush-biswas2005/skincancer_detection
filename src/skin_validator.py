"""
Multi-stage image validation and skin detection pipeline for DermScan.

Pipeline:
  1. Image File Integrity & Dimensions (PIL)
  2. Image Quality & Focus (Brightness, Exposure, Contrast, Blur via Laplacian variance)
  3. Human Skin Chromaticity & Region Analysis (YCbCr + RGB calibrated across Fitzpatrick phototypes I-VI)
  4. Deep Semantic Gatekeeper (MobileNetV3-Small ImageNet-1K: animals, vehicles, landscapes, objects, documents)
"""

import numpy as np
import torch
from PIL import Image, ImageStat, ImageFilter
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ValidationResult:
    """Represents the outcome of the image validation pipeline."""

    def __init__(self, is_valid: bool, stage: str, message: str, details: dict = None):
        self.is_valid = is_valid
        self.stage = stage          # 'integrity' | 'quality' | 'skin_region' | 'semantic' | 'passed'
        self.message = message
        self.details = details or {}

    def __repr__(self):
        return (f"ValidationResult(is_valid={self.is_valid}, stage='{self.stage}', "
                f"message='{self.message}')")


# ---------------------------------------------------------------------------
# Stage 1 & 2: Integrity & Image Quality Checkers
# ---------------------------------------------------------------------------

class ImageQualityChecker:
    """Checks file readability, dimensions, exposure, contrast, and focus."""

    MIN_DIMENSION = 64
    MAX_DIMENSION = 10000
    MIN_BRIGHTNESS = 22.0     # Mean channel value (0-255)
    MAX_BRIGHTNESS = 245.0
    MIN_STD_DEV = 8.0         # Catches flat/empty single-color images
    MIN_SHARPNESS = 30.0      # Laplacian variance threshold for blur (real skin lesions > 100)

    def check(self, image_path: str) -> ValidationResult:
        try:
            with Image.open(image_path) as img:
                img.verify()
            img = Image.open(image_path).convert('RGB')
        except Exception:
            return ValidationResult(
                is_valid=False,
                stage='integrity',
                message='Invalid Image! Please upload a valid PNG, JPG, or JPEG photograph of human skin.'
            )

        w, h = img.size

        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            return ValidationResult(
                is_valid=False,
                stage='quality',
                message=f'Invalid Image! Image resolution is too low ({w}×{h}px). '
                        f'Please upload a clear photograph of human skin with at least '
                        f'{self.MIN_DIMENSION}×{self.MIN_DIMENSION} pixels.',
                details={'width': w, 'height': h}
            )

        if w > self.MAX_DIMENSION or h > self.MAX_DIMENSION:
            return ValidationResult(
                is_valid=False,
                stage='quality',
                message='Invalid Image! Image resolution is unusually large. '
                        'Please resize the image and upload again.',
                details={'width': w, 'height': h}
            )

        stat = ImageStat.Stat(img)
        mean_brightness = sum(stat.mean) / 3.0
        std_brightness = sum(stat.stddev) / 3.0

        if mean_brightness < self.MIN_BRIGHTNESS:
            return ValidationResult(
                is_valid=False,
                stage='quality',
                message='Invalid Image! The photograph is too dark for accurate skin analysis. '
                        'Please upload a well-lit, clearly visible photograph of the skin.',
                details={'mean_brightness': round(mean_brightness, 1)}
            )

        if mean_brightness > self.MAX_BRIGHTNESS:
            return ValidationResult(
                is_valid=False,
                stage='quality',
                message='Invalid Image! The photograph is severely overexposed. '
                        'Please upload a properly exposed photograph of the skin.',
                details={'mean_brightness': round(mean_brightness, 1)}
            )

        if std_brightness < self.MIN_STD_DEV:
            return ValidationResult(
                is_valid=False,
                stage='quality',
                message='Invalid Image! The uploaded image lacks visual detail or contrast. '
                        'Please upload an authentic photograph of human skin.',
                details={'std_dev': round(std_brightness, 1)}
            )

        # Focus / Blur check using Laplacian variance
        gray = img.convert('L')
        laplacian_filter = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1,
                    -1,  8, -1,
                    -1, -1, -1],
            scale=1,
            offset=128
        )
        edge_image = gray.filter(laplacian_filter)
        edge_arr = np.array(edge_image, dtype=np.float32) - 128.0
        sharpness = float(np.var(edge_arr))

        if sharpness < self.MIN_SHARPNESS:
            return ValidationResult(
                is_valid=False,
                stage='quality',
                message='Invalid Image! The image appears too blurry or out of focus. '
                        'Please upload a sharp, focused photograph of human skin.',
                details={'sharpness': round(sharpness, 2)}
            )

        return ValidationResult(
            is_valid=True,
            stage='quality',
            message='Image quality checks passed.'
        )


# ---------------------------------------------------------------------------
# Stage 3: Human Skin Chromaticity & Region Analysis (YCbCr + RGB)
# ---------------------------------------------------------------------------

class SkinRegionValidator:
    """
    Evaluates skin-tone chromaticity across diverse human phototypes (Fitzpatrick I-VI)
    without relying on OpenCV. Uses standard YCbCr and RGB boundary equations.
    """

    MIN_SKIN_PERCENTAGE = 18.0  # Macro/lesion skin images typically occupy > 35%

    def validate(self, image_path: str) -> ValidationResult:
        try:
            img = Image.open(image_path).convert('RGB')
            # Downsample for fast, stable chromaticity statistics
            img_small = img.resize((320, int(320 * img.height / img.width)), Image.Resampling.BILINEAR)
            arr = np.array(img_small, dtype=np.float32)
        except Exception:
            return ValidationResult(
                is_valid=False,
                stage='skin_region',
                message='Invalid Image! Could not process image color channels.'
            )

        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]

        # YCbCr conversion (ITU-R BT.601)
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
        Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0

        # Human skin condition covering light, olive, brown, and dark skin tones:
        # 1. Red channel dominates green & blue (hemoglobin/melanin absorption profile)
        rgb_skin = (R > G) & (G >= B) & ((R - G) >= 6.0) & ((R - B) >= 10.0)

        # 2. Compact chromatic cluster in YCbCr space
        ycbcr_skin = (Cb >= 75.0) & (Cb <= 140.0) & (Cr >= 128.0) & (Cr <= 185.0) & (Cr > Cb)

        skin_mask = rgb_skin & ycbcr_skin
        skin_pct = float(np.mean(skin_mask) * 100.0)

        # Vegetation / foliage check (strong green dominance)
        green_foliage_pct = float(np.mean((G > R + 15.0) & (G > B + 10.0)) * 100.0)
        if green_foliage_pct > 25.0:
            return ValidationResult(
                is_valid=False,
                stage='skin_region',
                message='Invalid Image! Landscape, plant, or outdoor scenery detected. '
                        'Please upload a clear photograph of human skin for accurate analysis.',
                details={'foliage_pct': round(green_foliage_pct, 1)}
            )

        # Ocean / sky check (strong blue dominance)
        blue_sky_pct = float(np.mean((B > R + 20.0) & (B > G + 10.0)) * 100.0)
        if blue_sky_pct > 25.0:
            return ValidationResult(
                is_valid=False,
                stage='skin_region',
                message='Invalid Image! Sky, water, or non-skin scene detected. '
                        'Please upload a clear photograph of human skin for accurate analysis.',
                details={'sky_pct': round(blue_sky_pct, 1)}
            )

        # Pure monochrome / grayscale check (document, b&w screenshot)
        chroma_spread = float(np.mean(np.abs(R - G) + np.abs(G - B)))
        if chroma_spread < 6.0:
            return ValidationResult(
                is_valid=False,
                stage='skin_region',
                message='Invalid Image! Monochromatic, screenshot, or document image detected. '
                        'Please upload an authentic color photograph of human skin.',
                details={'chroma_spread': round(chroma_spread, 2)}
            )

        if skin_pct < self.MIN_SKIN_PERCENTAGE:
            return ValidationResult(
                is_valid=False,
                stage='skin_region',
                message='Invalid Image! Insufficient human skin detected in the image. '
                        'Please upload a clear, focused photograph of human skin for accurate analysis.',
                details={'skin_percentage': round(skin_pct, 1)}
            )

        return ValidationResult(
            is_valid=True,
            stage='skin_region',
            message='Skin region chromaticity check passed.',
            details={'skin_percentage': round(skin_pct, 1)}
        )


# ---------------------------------------------------------------------------
# Stage 4: Deep Semantic Gatekeeper (MobileNetV3 ImageNet-1K)
# ---------------------------------------------------------------------------

class SemanticGatekeeper:
    """
    Semantic gatekeeper powered by MobileNetV3-Small (ImageNet-1K).
    Detects animals, vehicles, electronics, furniture, food, documents, and scenery.
    """

    # Animal classes in ImageNet: 0 to 397
    _ANIMAL_END = 398

    # Medical & dermatoscopy classes that MUST NOT be rejected
    # (In ImageNet, skin lesions often match 'tick', 'Band Aid', 'ruler', 'safety pin')
    _ALLOWED_MEDICAL = {
        78,   # tick (dermoscopic moles & carcinomas resemble small spots)
        419,  # Band Aid (adhesive bandage on skin)
        769,  # rule, ruler (medical calibration scale in dermatoscopy)
        771,  # safety pin (clinical marker/scale)
        838,  # sunscreen (dermatological application)
        843,  # syringe (medical injection)
        682,  # nipple (anatomical landmark)
    }

    _NON_SKIN_KEYWORDS = [
        # Vehicles & Aviation
        'car', 'bus', 'truck', 'trailer', 'tractor', 'jeep', 'limousine',
        'convertible', 'minivan', 'pickup', 'scooter', 'motorcycle', 'moped',
        'bicycle', 'tricycle', 'aircraft', 'airliner', 'warplane', 'helicopter',
        'boat', 'ship', 'canoe', 'kayak', 'submarine', 'yacht', 'speedboat',
        'locomotive', 'railcar', 'streetcar', 'forklift',

        # Electronics & Technology
        'laptop', 'computer', 'notebook', 'desktop', 'monitor', 'screen',
        'television', 'phone', 'cellular telephone', 'dial telephone',
        'keyboard', 'mouse', 'mouse, computer mouse', 'joystick', 'printer',
        'hard disc', 'modem', 'loudspeaker', 'microphone', 'radio', 'remote control',

        # Animals & Pets (indices >= 398)
        'teddy', 'teddy bear',

        # Landscapes & Scenery
        'alp', 'mountain', 'volcano', 'valley', 'cliff', 'promontory',
        'sandbar', 'coral reef', 'lakeside', 'seashore', 'geyser',

        # Food & Beverages
        'pizza', 'hamburger', 'cheeseburger', 'hotdog', 'burrito', 'sandwich',
        'bagel', 'pretzel', 'banana', 'orange', 'lemon', 'pineapple', 'apple',
        'strawberry', 'broccoli', 'cauliflower', 'cabbage', 'cucumber',
        'acorn squash', 'butternut squash', 'bell pepper', 'mushroom',
        'ice cream', 'ice lolly', 'french loaf', 'espresso', 'beer bottle',
        'wine bottle', 'cocktail shaker',

        # Furniture & Appliances
        'chair', 'table', 'desk', 'couch', 'sofa', 'wardrobe', 'bookcase',
        'refrigerator', 'washer', 'dishwasher', 'toilet', 'bathtub', 'microwave',

        # Documents & Office
        'web site', 'website', 'menu', 'newspaper', 'comic book', 'book jacket',
        'crossword puzzle', 'envelope', 'binder', 'pencil box', 'rubber eraser',

        # Clothing & Accessories (when dominant subject)
        'vestment', 'academic gown', 'suit', 'military uniform', 'kimono',
        'trench coat', 'fur coat', 'sombrero', 'cowboy hat', 'mortarboard',
        'running shoe', 'boot', 'sandal', 'seat belt'
    ]

    def __init__(self, device):
        self.device = device
        self.model = None
        self.transform = None
        self.categories = []
        self.reject_indices = set()
        self.is_ready = False

        try:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.model = mobilenet_v3_small(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.transform = weights.transforms()
            self.categories = weights.meta["categories"]

            # Animals (0 to 397, excluding allowed medical surrogates)
            for idx in range(0, self._ANIMAL_END):
                if idx not in self._ALLOWED_MEDICAL:
                    self.reject_indices.add(idx)

            # Keyword matching for other non-skin classes
            kw_set = [k.lower() for k in self._NON_SKIN_KEYWORDS]
            for idx in range(self._ANIMAL_END, len(self.categories)):
                if idx in self._ALLOWED_MEDICAL:
                    continue
                name = self.categories[idx].lower()
                for kw in kw_set:
                    if kw in name:
                        self.reject_indices.add(idx)
                        break

            self.is_ready = True
            print(f"[Gatekeeper] Semantic gatekeeper ready ({len(self.reject_indices)} non-skin classes indexed).")
        except Exception as e:
            print(f"[Gatekeeper] Warning: Could not initialize semantic gatekeeper: {e}")

    def validate(self, image_path: str) -> ValidationResult:
        if not self.is_ready:
            # If gatekeeper failed to load, pass through to avoid breaking system
            return ValidationResult(is_valid=True, stage='semantic', message='Gatekeeper bypassed (offline mode).')

        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output[0], dim=0)

            top5_probs, top5_indices = torch.topk(probs, 5)
            top1_idx = top5_indices[0].item()
            top1_prob = top5_probs[0].item()
            top1_name = self.categories[top1_idx]

            # 1. Medical / Dermatoscopy whitelisted proxy classes: ALWAYS ALLOW
            if top1_idx in self._ALLOWED_MEDICAL:
                return ValidationResult(
                    is_valid=True,
                    stage='semantic',
                    message='Dermatological specimen pattern detected.',
                    details={'detected_pattern': top1_name, 'confidence': round(top1_prob, 3)}
                )

            # 2. Check if top-1 is an animal
            if top1_idx < self._ANIMAL_END and top1_prob > 0.12:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    message='Invalid Image! Animal detected. '
                            'Please upload a clear photograph of human skin for accurate analysis.',
                    details={'subject': top1_name, 'confidence': f'{top1_prob * 100:.1f}%'}
                )

            # 3. Check cumulative animal probability
            animal_prob = sum(
                top5_probs[i].item()
                for i in range(5)
                if top5_indices[i].item() < self._ANIMAL_END and top5_indices[i].item() not in self._ALLOWED_MEDICAL
            )
            if animal_prob > 0.25:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    message='Invalid Image! Animal photograph detected. '
                            'Please upload a clear photograph of human skin for accurate analysis.',
                    details={'animal_confidence': f'{animal_prob * 100:.1f}%'}
                )

            # 4. Check if top-1 is in the non-skin reject set (objects, scenery, food, etc.)
            if top1_idx in self.reject_indices and top1_prob > 0.15:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    message='Invalid Image! Please upload a clear photograph of human skin for accurate analysis.',
                    details={'detected_subject': top1_name, 'confidence': f'{top1_prob * 100:.1f}%'}
                )

            # 5. Check cumulative non-skin probability
            reject_prob = sum(
                top5_probs[i].item()
                for i in range(5)
                if top5_indices[i].item() in self.reject_indices
            )
            if reject_prob > 0.40:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    message='Invalid Image! Please upload a clear photograph of human skin for accurate analysis.',
                    details={'non_skin_confidence': f'{reject_prob * 100:.1f}%'}
                )

            return ValidationResult(
                is_valid=True,
                stage='semantic',
                message='Semantic gatekeeper passed.',
                details={'top1': top1_name, 'confidence': f'{top1_prob * 100:.1f}%'}
            )

        except Exception as e:
            print(f"[Gatekeeper] Validation exception: {e}")
            return ValidationResult(
                is_valid=False,
                stage='semantic',
                message='Invalid Image! Error analyzing image content. Please upload a clear skin photograph.'
            )


# ---------------------------------------------------------------------------
# Integrated Multi-Stage Pipeline Orchestrator
# ---------------------------------------------------------------------------

class ValidationPipeline:
    """
    Orchestrates the full 4-stage validation pipeline:
      Stage 1: Image Format & Integrity
      Stage 2: Image Quality (Brightness, Contrast, Blur)
      Stage 3: Skin Region & Chromaticity (YCbCr + RGB)
      Stage 4: Semantic Gatekeeper (MobileNetV3 ImageNet)
    """

    def __init__(self, device):
        self.quality_checker = ImageQualityChecker()
        self.skin_validator = SkinRegionValidator()
        self.semantic_gatekeeper = SemanticGatekeeper(device)

    def validate(self, image_path: str) -> ValidationResult:
        # Stage 1 & 2: Integrity and Quality
        q_res = self.quality_checker.check(image_path)
        if not q_res.is_valid:
            return q_res

        # Stage 3: Skin Region Chromaticity
        s_res = self.skin_validator.validate(image_path)
        if not s_res.is_valid:
            return s_res

        # Stage 4: Semantic Content Gatekeeper
        sem_res = self.semantic_gatekeeper.validate(image_path)
        if not sem_res.is_valid:
            return sem_res

        return ValidationResult(
            is_valid=True,
            stage='passed',
            message='All validation checks passed successfully.'
        )
