"""
NSFW Guard Nodes for ComfyUI (Viddexa variant)

Uses HF moderation models with selectable repository:
- viddexa/nsfw-detection-2-nano
- viddexa/nsfw-detection-2-mini

Blocking policy:
- Block labels: porn, hentai, sexy
- Always pass: drawing
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

import folder_paths
from nodes import interrupt_processing
from server import PromptServer

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

try:
    from moderators import AutoModerator
except Exception:  # pragma: no cover
    AutoModerator = None

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModelForImageClassification = None


MODEL_OPTIONS = [
    "viddexa/nsfw-detection-2-nano",
    "viddexa/nsfw-detection-2-mini",
]
BLOCKED_LABEL_KEYWORDS = ("porn", "hentai", "sexy")
PASS_LABEL_KEYWORDS = ("safe", "drawing")


class NSFWContentError(Exception):
    """Custom exception for NSFW content detection."""

    def __init__(self, prediction: str, confidence: float):
        self.error_type = "nsfw_content_detected"
        self.prediction = prediction
        self.confidence = confidence
        message = (
            "NSFW content detected - workflow interrupted "
            f"(prediction: {prediction}, confidence: {confidence:.2%})"
        )
        super().__init__(message)

    def to_dict(self):
        return {
            "error": {
                "type": self.error_type,
                "message": str(self),
                "details": {
                    "prediction": self.prediction,
                    "confidence": self.confidence,
                },
            }
        }


def _label_contains_any(label: str, keywords: Tuple[str, ...]) -> bool:
    l = label.lower()
    return any(k in l for k in keywords)


def _id2label_to_dict(id2label) -> Dict[int, str]:
    if isinstance(id2label, dict):
        out = {}
        for k, v in id2label.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def _score_from_any(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _collect_label_scores_from_result(obj) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []

    def walk(x):
        if hasattr(x, "model_dump"):
            try:
                x = x.model_dump()
            except Exception:
                pass
        if hasattr(x, "to_dict"):
            try:
                x = x.to_dict()
            except Exception:
                pass

        if isinstance(x, dict):
            # Common single-prediction shape
            if "label" in x and any(k in x for k in ("score", "confidence", "probability", "prob")):
                label = str(x.get("label", ""))
                score = _score_from_any(
                    x.get("score", x.get("confidence", x.get("probability", x.get("prob", 0.0))))
                )
                items.append((label, score))

            # Common list-of-predictions shape
            if isinstance(x.get("predictions"), list):
                for p in x["predictions"]:
                    walk(p)

            # Numeric map shape: {"safe": 0.8, "sexy": 0.2} or {"f_sexy": 0.4}
            numeric_pairs = []
            for k, v in x.items():
                if isinstance(v, (int, float)):
                    lk = str(k).lower()
                    if any(tok in lk for tok in ("safe", "drawing", "porn", "hentai", "sexy")):
                        numeric_pairs.append((str(k), float(v)))
            if numeric_pairs:
                items.extend(numeric_pairs)

            # Recurse all values for nested shapes
            for v in x.values():
                if isinstance(v, (dict, list, tuple)):
                    walk(v)

        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)

    # Deduplicate by max score for each normalized label
    by_label: Dict[str, float] = {}
    for label, score in items:
        key = label.strip()
        if not key:
            continue
        by_label[key] = max(by_label.get(key, 0.0), float(score))

    items = [(k, v) for k, v in by_label.items()]
    return items


def _policy_decision(label_scores: List[Tuple[str, float]]) -> Tuple[bool, float, str]:
    if not label_scores:
        return False, 0.0, ""

    cleaned = [(str(label), float(score)) for label, score in label_scores]
    top_label, top_score = max(cleaned, key=lambda x: x[1])
    top_label_l = top_label.lower().strip()

    # Hard-pass classes requested by user
    if _label_contains_any(top_label_l, PASS_LABEL_KEYWORDS):
        return False, float(top_score), top_label

    # Hard-block classes requested by user
    if _label_contains_any(top_label_l, BLOCKED_LABEL_KEYWORDS):
        return True, float(top_score), top_label

    # Safety fallback: if any blocked label has score >= safe/drawing score, block.
    max_block = ("", 0.0)
    max_pass = ("", 0.0)
    for label, score in cleaned:
        ll = label.lower().strip()
        if _label_contains_any(ll, BLOCKED_LABEL_KEYWORDS) and score > max_block[1]:
            max_block = (label, float(score))
        if _label_contains_any(ll, PASS_LABEL_KEYWORDS) and score > max_pass[1]:
            max_pass = (label, float(score))
    if max_block[1] > 0.0 and max_block[1] >= max_pass[1]:
        return True, max_block[1], max_block[0]

    return False, float(top_score), top_label


class NSFWCheck:
    """
    Checks images for NSFW content.
    Supports model selection and blocks based on porn/hentai/sexy.
    """

    _cache: Dict[str, Dict] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_repo": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "check_nsfw"
    CATEGORY = "safety"

    def _ensure_dependencies(self):
        if AutoModerator is not None:
            return
        if snapshot_download is not None and AutoImageProcessor is not None and AutoModelForImageClassification is not None:
            return
        raise RuntimeError(
            "Missing dependencies. Preferred: pip install moderators. "
            "Fallback: pip install huggingface_hub transformers safetensors"
        )

    def _ensure_model(self, model_repo: str):
        self._ensure_dependencies()
        if model_repo in NSFWCheck._cache:
            return

        model_folder_name = model_repo.replace("/", "_")
        local_root = os.path.join(folder_paths.models_dir, "nsfw", model_folder_name)
        os.makedirs(local_root, exist_ok=True)

        if AutoModerator is not None:
            print(f"[NSFW Guard] Loading model via moderators from {model_repo}...")
            moderator = AutoModerator.from_pretrained(model_repo)
            NSFWCheck._cache[model_repo] = {
                "backend": "moderators",
                "model": moderator,
                "processor": None,
                "device": "cpu",
                "labels": {},
            }
            return

        if snapshot_download is None:
            raise RuntimeError("moderators unavailable and snapshot_download unavailable for fallback.")

        model_dir = snapshot_download(
            repo_id=model_repo,
            local_dir=local_root,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
        print(f"[NSFW Guard] Loading model (transformers fallback) from {model_dir}...")

        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        NSFWCheck._cache[model_repo] = {
            "backend": "transformers",
            "model": model,
            "processor": processor,
            "device": device,
            "labels": _id2label_to_dict(getattr(model.config, "id2label", {})),
        }

    def _predict_label_scores_transformers(
        self, model, processor, device: str, labels: Dict[int, str], pil_image: Image.Image
    ) -> List[Tuple[str, float]]:
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy().astype(float)

        label_scores: List[Tuple[str, float]] = []
        for idx, score in enumerate(probs):
            label_scores.append((labels.get(idx, str(idx)), float(score)))
        return label_scores

    def _predict_label_scores_moderators(self, moderator, pil_image: Image.Image) -> List[Tuple[str, float]]:
        result = None
        errs = []
        for inp in (pil_image, np.array(pil_image), {"image": pil_image}):
            try:
                result = moderator(inp)
                break
            except Exception as e:
                errs.append(str(e))
        if result is None:
            raise RuntimeError(f"moderators inference failed: {' | '.join(errs)}")

        # Official moderators pattern:
        # probs = {k: v for r in results for k, v in r.classifications.items()}
        probs: Dict[str, float] = {}
        for r in result:
            cls_map = getattr(r, "classifications", None)
            if isinstance(cls_map, dict):
                for k, v in cls_map.items():
                    key = str(k).strip()
                    score = _score_from_any(v)
                    probs[key] = max(probs.get(key, 0.0), score)

        label_scores = [(k, v) for k, v in probs.items()]
        print(f"[NSFW Guard] moderators raw={result}")
        print(f"[NSFW Guard] parsed label_scores={label_scores}")
        return label_scores

    def _predict_label_scores(self, model_bundle: Dict, pil_image: Image.Image) -> List[Tuple[str, float]]:
        backend = model_bundle["backend"]
        if backend == "moderators":
            return self._predict_label_scores_moderators(model_bundle["model"], pil_image)
        return self._predict_label_scores_transformers(
                model_bundle["model"],
                model_bundle["processor"],
                model_bundle["device"],
                model_bundle["labels"],
                pil_image,
            )

    def _raise_block(self, blocked_label: str, confidence: float):
        prediction = blocked_label or "nsfw"
        error = NSFWContentError(
            prediction=prediction,
            confidence=confidence,
        )

        PromptServer.instance.send_sync(
            "nsfw_guard.content_blocked",
            {
                "type": "nsfw_content_detected",
                "prediction": prediction,
                "confidence": confidence,
                "message": str(error),
            },
        )

        interrupt_processing(True)
        raise Exception(json.dumps(error.to_dict()))

    def check_nsfw(self, image: torch.Tensor, model_repo: str):
        self._ensure_model(model_repo)
        model_bundle = NSFWCheck._cache[model_repo]

        max_block_conf = 0.0
        max_block_label = ""

        for i in range(image.shape[0]):
            img_np = np.clip(image[i].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np, mode="RGB")

            label_scores = self._predict_label_scores(model_bundle, pil_image)
            should_block, conf, label = _policy_decision(label_scores)

            if conf > max_block_conf:
                max_block_conf = conf
                max_block_label = label

            if should_block:
                self._raise_block(max_block_label, max_block_conf)

        return (image,)


class NSFWLoadModel:
    """Loads NSFW model once and outputs a shared model bundle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
            }
        }

    RETURN_TYPES = ("NSFW_GUARD_MODEL",)
    RETURN_NAMES = ("nsfw_model",)
    FUNCTION = "load_model"
    CATEGORY = "safety"

    def load_model(self, model_repo: str):
        checker = NSFWCheck()
        checker._ensure_model(model_repo)
        return ((model_repo, NSFWCheck._cache[model_repo]),)


class NSFWCheckWithModel:
    """Checks images for NSFW content using preloaded model from NSFWLoadModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nsfw_model": ("NSFW_GUARD_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "check_nsfw"
    CATEGORY = "safety"

    def check_nsfw(self, nsfw_model, image: torch.Tensor):
        if not isinstance(nsfw_model, tuple) or len(nsfw_model) != 2:
            raise RuntimeError("Invalid NSFW_GUARD_MODEL. Use NSFW Load Model (HF) output.")

        _, model_bundle = nsfw_model
        if not isinstance(model_bundle, dict) or "backend" not in model_bundle:
            raise RuntimeError("Invalid NSFW_GUARD_MODEL payload.")

        checker = NSFWCheck()
        max_block_conf = 0.0
        max_block_label = ""

        for i in range(image.shape[0]):
            img_np = np.clip(image[i].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np, mode="RGB")

            label_scores = checker._predict_label_scores(model_bundle, pil_image)
            should_block, conf, label = _policy_decision(label_scores)

            if conf > max_block_conf:
                max_block_conf = conf
                max_block_label = label

            if should_block:
                checker._raise_block(max_block_label, max_block_conf)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "NSFWCheck": NSFWCheck,
    "NSFWLoadModel": NSFWLoadModel,
    "NSFWCheckWithModel": NSFWCheckWithModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWCheck": "NSFW Check (HF Classifier)",
    "NSFWLoadModel": "NSFW Load Model (HF)",
    "NSFWCheckWithModel": "NSFW Check (HF, Shared Model)",
}
