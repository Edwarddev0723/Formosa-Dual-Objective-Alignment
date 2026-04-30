"""Tests for Transformers compatibility helpers in backbone loading."""
from __future__ import annotations

from types import SimpleNamespace

from formosa_dual.config.schema import ModelConfig
from formosa_dual.models.backbone import _from_pretrained_compat, resolve_vision_lm_model_class


def test_resolve_qwen25_class():
    cls = resolve_vision_lm_model_class("Qwen/Qwen2.5-VL-7B-Instruct")
    assert cls.__name__ == "Qwen2_5_VLForConditionalGeneration"


def test_from_pretrained_compat_retries_torch_dtype():
    calls = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(kwargs)
            if "dtype" in kwargs:
                raise TypeError("unexpected keyword argument 'dtype'")
            return SimpleNamespace(name=name, kwargs=kwargs)

    cfg = ModelConfig(
        name="dummy/model",
        torch_dtype="bf16",
        attn_implementation="sdpa",
    )
    model = _from_pretrained_compat(DummyModel, cfg, "bf16")

    assert model.name == "dummy/model"
    assert "dtype" in calls[0]
    assert calls[1]["torch_dtype"] == "bf16"
