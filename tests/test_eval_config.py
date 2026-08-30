import argparse
import importlib.util
from pathlib import Path


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "eval_recall.py"
    spec = importlib.util.spec_from_file_location("eval_recall_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_config_values_are_used_when_cli_is_absent(tmp_path):
    mod = _load_eval_module()
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        "ratios: [0.01, 0.03]\n"
        "topk: [9]\n"
        "hit_half_size: 24\n"
        "decoder:\n"
        "  kernel_size: 3\n"
        "  score_threshold: 0.25\n",
        encoding="utf-8",
    )
    loaded = mod.load_eval_config(str(cfg))
    args = argparse.Namespace(ratios=None, topk=None, kernel_size=None, score_threshold=None, hit_half_size=None)
    settings = mod.resolve_eval_settings(args, loaded)
    assert settings["ratios"] == [0.01, 0.03]
    assert settings["topk"] == [9]
    assert settings["kernel_size"] == 3
    assert settings["score_threshold"] == 0.25
    assert settings["hit_half_size"] == 24


def test_eval_cli_values_override_config(tmp_path):
    mod = _load_eval_module()
    cfg = {"ratios": [0.01], "topk": [9], "hit_half_size": 32, "decoder": {"kernel_size": 7, "score_threshold": 0.0}}
    args = argparse.Namespace(ratios="0.02", topk="4,16", kernel_size=1, score_threshold=0.5, hit_half_size=12)
    settings = mod.resolve_eval_settings(args, cfg)
    assert settings["ratios"] == [0.02]
    assert settings["topk"] == [4, 16]
    assert settings["kernel_size"] == 1
    assert settings["score_threshold"] == 0.5
    assert settings["hit_half_size"] == 12
