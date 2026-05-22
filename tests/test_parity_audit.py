from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_parity.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("sparsevideo_audit_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(method: str, tmp_path: Path, *, steps: int = 50) -> dict:
    output = tmp_path / f"{method}.mp4"
    output.write_bytes(b"fake mp4")
    return {
        "method": method,
        "model_arg": "wan1.3b",
        "status": "ok",
        "strict_kernels": True,
        "allow_debug_fallbacks": False,
        "num_inference_steps": steps,
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "output_file": str(output),
        "preflight": {"errors": [], "warnings": []},
        "timings": {"generate_sec": 10.0, "total_sec": 12.0},
        "sparse_attention_handle": {
            "method_runtime": {
                "dispatch_counts": {"sparse": 10} if method != "dense" else {"dense": 10},
                "backend_counts": {"native": 10} if method != "dense" else {},
            }
        },
    }


def _diffsynth_record(method: str, model: str, *, patched_count: int) -> dict:
    patched_paths = [f"dit.blocks.{idx}.self_attn.attn" for idx in range(patched_count)]
    apply_summary = {
        "pipeline_backend": "diffsynth",
        "diffsynth_version": "2.0.12",
        "model_key": model,
        "model_type": "wan",
        "num_self_attn_layers": max(patched_count, 1),
        "patched_attention_count": patched_count,
        "patched_attention_paths": patched_paths,
        "unpatched_attention_paths": [],
        "method_runtime": {"backend_counts": {}, "dispatch_counts": {}, "total_calls": 0},
    }
    restore_summary = dict(apply_summary)
    restore_summary["restored"] = True
    return {
        "backend": "diffsynth",
        "model": model,
        "method": method,
        "status": "ok",
        "mode": "apply_only",
        "resolved_model": {"model": model, "complete": True, "missing": []},
        "method_config": {},
        "timings": {"total_sec": 1.0},
        "elapsed_sec": 1.0,
        "cuda": {"device": "cuda", "available": True},
        "apply_summary": apply_summary,
        "restore_summary": restore_summary,
    }


def _mark_current_flashomni_policy(record: dict, audit_mod) -> dict:
    record["source_fingerprints"] = {
        "flashomni_policy_sha256": audit_mod._current_flashomni_policy_sha256(),
        "flashomni_method_sha256": audit_mod._current_flashomni_method_sha256(),
    }
    return record


def _patch_valid_video_artifacts(monkeypatch, audit_mod):
    def valid_artifact(record):
        return {
            "path": str(record.get("output_file")),
            "exists": True,
            "warnings": [],
            "actual_frames": int(record.get("num_frames") or 1),
            "ffprobe": {
                "width": str(record.get("width") or 0),
                "height": str(record.get("height") or 0),
                "nb_frames": str(record.get("num_frames") or 1),
                "duration": "1.0",
                "avg_frame_rate": "1/1",
                "bit_rate": "1000000",
            },
        }

    monkeypatch.setattr(audit_mod, "_video_artifact_qc", valid_artifact)


def test_audit_reports_missing_quality_dispatch_for_sparse_methods(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    records = [_record("dense", tmp_path), _record("svg2", tmp_path, steps=6)]

    audit = audit_mod.build_audit(records, min_steps=50)

    assert audit["overall_status"] == "incomplete"
    gates = {item["gate"]: item for item in audit["checklist"]}
    assert gates["public_api_and_config_contract"]["status"] == "pass"
    assert "apply" in gates["public_api_and_config_contract"]["evidence"]["public_api"]
    assert gates["all_backbone_support_contract"]["status"] == "pass"
    backbone_evidence = gates["all_backbone_support_contract"]["evidence"]
    assert backbone_evidence["models"]["cogvideox-t2v"]["sparse_methods"] is None
    assert backbone_evidence["limited_methods_by_model_type"] == {}
    assert backbone_evidence["models"]["sana-video"]["compatibility_label"] == "incompatible"
    assert backbone_evidence["models"]["kandinsky5-t2v"]["compatibility_label"] == "native-N/A"
    assert backbone_evidence["aliases"]["ltx-i2v"] == "ltx-video-i2v"
    assert (
        backbone_evidence["processor_classes"]["mochi"]
        == "sparsevideo.processors.mochi.SparseMochiAttnProcessor"
    )
    assert audit["methods"]["dense"]["status"] == "pass"
    assert audit["methods"]["svg2"]["status"] == "partial"
    assert "needs a 50-step mp4 quality record" in audit["methods"]["svg2"]["missing"]


def test_all_backbone_smoke_gate_counts_latent_smoke_and_aliases(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(
        audit_mod,
        "EXPECTED_BACKBONE_SMOKE_METHODS",
        {"cogvideox-t2v": ("dense", "svg2")},
    )
    dry_run = _record("dense", tmp_path, steps=1)
    dry_run["model_arg"] = "cogvideox"
    dry_run["status"] = "dry_run"

    dense = _record("dense", tmp_path, steps=1)
    dense["model_arg"] = "cogvideox"
    dense["skip_decode"] = True
    dense["strict_kernels"] = False
    dense["allow_debug_fallbacks"] = True

    svg2 = _record("svg2", tmp_path, steps=1)
    svg2["model_arg"] = "cog"
    svg2["skip_decode"] = True
    svg2["strict_kernels"] = False
    svg2["allow_debug_fallbacks"] = True
    svg2["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {"flashinfer": 41}
    svg2["sparse_attention_handle"]["method_runtime"]["dispatch_counts"] = {
        "dense": 1,
        "sparse": 41,
    }

    gate = audit_mod._all_backbone_smoke_evidence_gate([dry_run, dense, svg2])

    assert gate["status"] == "pass"
    assert gate["missing"] == []
    methods = gate["evidence"]["models"]["cogvideox-t2v"]["methods"]
    assert methods["dense"]["status"] == "ok"
    assert methods["svg2"]["backend_counts"] == {"flashinfer": 41}
    assert methods["svg2"]["strict_kernels"] is False


def test_new_backbone_smoke_gate_requires_all_public_sparse_methods():
    audit_mod = _load_audit_module()

    methods = audit_mod.EXPECTED_BACKBONE_SMOKE_METHODS["cogvideox-t2v"]

    assert methods == audit_mod.EXPECTED_FULL_BACKBONE_METHODS
    assert set(methods) == set(audit_mod.EXPECTED_METHODS)


def test_default_audit_metrics_include_diffsynth_smoke_outputs():
    audit_mod = _load_audit_module()

    assert audit_mod.DEFAULT_DIFFSYNTH_METRICS_GLOB == "outputs/diffsynth_method_smoke/**/*.jsonl"
    assert audit_mod.DEFAULT_DIFFSYNTH_METRICS_GLOB in audit_mod.DEFAULT_METRICS_GLOBS


def test_diffsynth_apply_restore_gate_counts_all_method_apply_only_records(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    records = [
        _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0),
        _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=30),
    ]

    gate = audit_mod._diffsynth_apply_restore_evidence_gate(records)

    assert gate["status"] == "pass"
    assert gate["missing"] == []
    methods = gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]
    assert methods["dense"]["patched_attention_count"] == 0
    assert methods["svg2"]["patched_attention_count"] == 30
    assert methods["svg2"]["diffsynth_version"] == "2.0.12"
    assert methods["svg2"]["restored"] is True
    assert methods["svg2"]["method_config"] == {}
    assert methods["svg2"]["timings"] == {"total_sec": 1.0}
    assert methods["svg2"]["cuda"] == {"device": "cuda", "available": True}


def test_diffsynth_checkpoint_gate_reports_incomplete_bundles(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(
        audit_mod,
        "_diffsynth_checkpoint_records",
        lambda: (
            "/models",
            [
                {
                    "model": "wan21-t2v-1.3b",
                    "family": "wan",
                    "pipeline": "WanVideoPipeline",
                    "required_inputs": [],
                    "complete": True,
                    "components": {"dit": ["/models/Wan2.1-T2V-1.3B/model.safetensors"]},
                    "missing": [],
                },
                {
                    "model": "wan21-vace-1.3b",
                    "family": "wan",
                    "pipeline": "WanVideoPipeline",
                    "required_inputs": [],
                    "complete": False,
                    "components": {},
                    "missing": ["dit: missing files"],
                },
            ],
        ),
    )
    monkeypatch.setattr(
        audit_mod,
        "_diffsynth_deferred_model_records",
        lambda: [
            {
                "model": "wan22-dancer-14b",
                "family": "wan",
                "pipeline": "WanVideoPipeline",
                "origin_repo": "Wan-AI/Wan2.2-Dancer-14B",
                "origin_pattern": "global_model.safetensors",
                "required_inputs": ["wantodance_music_path"],
                "deferred_reason": "local-only until source is confirmed",
            }
        ],
    )

    gate = audit_mod._diffsynth_checkpoint_availability_gate()

    assert gate["status"] == "fail"
    assert gate["evidence"]["model_root"] == "/models"
    assert gate["evidence"]["complete_count"] == 1
    assert gate["evidence"]["incomplete_count"] == 1
    assert gate["evidence"]["missing_component_checks"] == 1
    assert gate["evidence"]["deferred_count"] == 1
    assert gate["evidence"]["deferred_models"]["wan22-dancer-14b"]["origin_repo"] == "Wan-AI/Wan2.2-Dancer-14B"
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["complete"] is True
    assert gate["evidence"]["models"]["wan21-vace-1.3b"]["missing_count"] == 1
    assert "wan21-vace-1.3b: incomplete DiffSynth bundle (1 missing component checks)" in gate["missing"]
    assert not any("wan22-dancer-14b" in item for item in gate["missing"])


def test_diffsynth_checkpoint_gate_does_not_fail_on_deferred_models(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(
        audit_mod,
        "_diffsynth_checkpoint_records",
        lambda: (
            "/models",
            [
                {
                    "model": "wan21-t2v-1.3b",
                    "family": "wan",
                    "pipeline": "WanVideoPipeline",
                    "required_inputs": [],
                    "complete": True,
                    "components": {"dit": ["/models/Wan2.1-T2V-1.3B/model.safetensors"]},
                    "missing": [],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        audit_mod,
        "_diffsynth_deferred_model_records",
        lambda: [
            {
                "model": "wan22-dancer-14b",
                "family": "wan",
                "pipeline": "WanVideoPipeline",
                "origin_repo": "Wan-AI/Wan2.2-Dancer-14B",
                "origin_pattern": "global_model.safetensors",
                "required_inputs": ["wantodance_music_path"],
                "deferred_reason": "local-only until source is confirmed",
            }
        ],
    )

    gate = audit_mod._diffsynth_checkpoint_availability_gate()

    assert gate["status"] == "pass"
    assert gate["missing"] == []
    assert gate["evidence"]["complete_count"] == 1
    assert gate["evidence"]["incomplete_count"] == 0
    assert gate["evidence"]["deferred_count"] == 1
    assert "wan22-dancer-14b" in gate["evidence"]["deferred_models"]


def test_diffsynth_apply_restore_gate_rejects_missing_restore_or_patch(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=0)
    sparse["restore_summary"]["restored"] = False

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["dense"]["restored"] is True
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["svg2"] is None


def test_diffsynth_apply_restore_gate_rejects_partial_attention_patch(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=1)
    sparse["apply_summary"]["num_self_attn_layers"] = 2
    sparse["restore_summary"]["num_self_attn_layers"] = 2

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["svg2"] is None


def test_diffsynth_apply_restore_gate_rejects_missing_layer_count(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=30)
    dense["apply_summary"].pop("num_self_attn_layers")
    sparse["apply_summary"].pop("num_self_attn_layers")

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/dense: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]


def test_diffsynth_apply_restore_gate_rejects_non_evidence_mode(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=30)
    dense["mode"] = "dry_run"
    sparse["mode"] = "dry_run"

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/dense: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]


def test_diffsynth_apply_restore_gate_requires_sparse_dispatch_for_generate(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=30)
    sparse["mode"] = "generate"
    sparse["generate_summary"] = {
        **sparse["apply_summary"],
        "method_runtime": {"backend_counts": {"flashinfer": 30}, "dispatch_counts": {"dense": 30}},
    }

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["dense"]["restored"] is True
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["svg2"] is None

    sparse["generate_summary"]["method_runtime"]["dispatch_counts"] = {"sparse": 60}
    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "pass"
    assert gate["missing"] == []
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["svg2"]["dispatch_counts"] == {"sparse": 60}


def test_diffsynth_apply_restore_gate_rejects_incomplete_resolved_model(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=30)
    dense["resolved_model"] = {"complete": False, "missing": ["dit: missing files"]}
    sparse["resolved_model"] = {"complete": False, "missing": ["dit: missing files"]}

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/dense: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]


def test_diffsynth_apply_restore_gate_requires_resolved_model_metadata(monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(audit_mod, "_expected_diffsynth_model_keys", lambda: ("wan21-t2v-1.3b",))
    monkeypatch.setattr(audit_mod, "EXPECTED_DIFFSYNTH_APPLY_METHODS", ("dense", "svg2"))
    dense = _diffsynth_record("dense", "wan21-t2v-1.3b", patched_count=0)
    sparse = _diffsynth_record("svg2", "wan21-t2v-1.3b", patched_count=30)
    dense.pop("resolved_model")
    sparse["resolved_model"] = {"model": "wan22-ti2v-5b", "complete": True, "missing": []}

    gate = audit_mod._diffsynth_apply_restore_evidence_gate([dense, sparse])

    assert gate["status"] == "fail"
    assert "wan21-t2v-1.3b/dense: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert "wan21-t2v-1.3b/svg2: needs status=ok DiffSynth apply/restore record" in gate["missing"]
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["dense"] is None
    assert gate["evidence"]["models"]["wan21-t2v-1.3b"]["methods"]["svg2"] is None


def test_all_backbone_smoke_gate_reports_missing_and_ignores_dry_run(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(
        audit_mod,
        "EXPECTED_BACKBONE_SMOKE_METHODS",
        {"mochi-1": ("dense", "svg2")},
    )
    dry_run = _record("dense", tmp_path, steps=1)
    dry_run["model_arg"] = "mochi"
    dry_run["status"] = "dry_run"

    gate = audit_mod._all_backbone_smoke_evidence_gate([dry_run])

    assert gate["status"] == "fail"
    assert gate["evidence"]["models"]["mochi-1"]["records_seen"] == 1
    assert "mochi-1/dense: needs >=1-step status=ok smoke record" in gate["missing"]
    assert "mochi-1/svg2: needs >=1-step status=ok smoke record" in gate["missing"]


def test_all_backbone_checkpoint_gate_reports_missing_index_refs(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    model_root = tmp_path / "models"
    transformer = model_root / "mochi-1" / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "present.safetensors").write_bytes(b"weights")
    (transformer / "model.safetensors.index.json").write_text(
        """
        {
          "metadata": {},
          "weight_map": {
            "a": "present.safetensors",
            "b": "missing.safetensors"
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit_mod,
        "EXPECTED_BACKBONE_SMOKE_METHODS",
        {"mochi-1": ("dense",)},
    )

    gate = audit_mod._all_backbone_checkpoint_availability_gate(model_root)

    assert gate["status"] == "fail"
    assert gate["evidence"]["models"]["mochi-1"]["status"] == "missing_index_refs"
    assert gate["evidence"]["models"]["mochi-1"]["missing_index_refs"] == [
        "transformer/missing.safetensors"
    ]
    assert "mochi-1: missing 1/2 indexed checkpoint files" in gate["missing"]


def test_all_backbone_checkpoint_gate_reports_missing_dir(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    monkeypatch.setattr(
        audit_mod,
        "EXPECTED_BACKBONE_SMOKE_METHODS",
        {"easyanimate-v5-t2v-12b": ("dense",)},
    )

    gate = audit_mod._all_backbone_checkpoint_availability_gate(tmp_path / "models")

    assert gate["status"] == "fail"
    assert gate["evidence"]["models"]["easyanimate-v5-t2v-12b"]["status"] == "missing_dir"
    assert "easyanimate-v5-t2v-12b: checkpoint status=missing_dir" in gate["missing"]


def test_flashomni_global_random_is_not_parity_evidence(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    record = _record("flashomni", tmp_path)
    record["method_config"] = {"sparse_pattern": "global_random"}
    record["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_global_random_upstream": 10
    }

    audit = audit_mod.build_audit([record], min_steps=50)

    assert audit["methods"]["flashomni"]["status"] == "partial"
    assert any("explicit sparse_info" in item for item in audit["methods"]["flashomni"]["missing"])


def test_flashomni_explicit_kernel_record_still_needs_video_policy(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    record = _record("flashomni", tmp_path)
    record["method_config"] = {"sparse_pattern": "explicit"}
    record["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_explicit_upstream": 10
    }

    audit = audit_mod.build_audit([record], min_steps=50)

    assert audit["methods"]["flashomni"]["status"] == "partial"
    assert audit["methods"]["flashomni"]["evidence"]["reference_policy"]["exists"] is True
    assert audit["methods"]["flashomni"]["evidence"]["reference_policy"]["video_policy_candidates"] == []
    git_history = audit["methods"]["flashomni"]["evidence"]["reference_policy"]["git_history"]
    assert git_history["available"] is True
    assert git_history["policy_candidates"] == []
    assert (
        "benchmark/test_attn_score.py"
        in audit["methods"]["flashomni"]["evidence"]["reference_policy"]["benchmark_score_sparse_helpers"]
    )
    assert (
        "benchmark/utils.py"
        in audit["methods"]["flashomni"]["evidence"]["reference_policy"]["benchmark_global_random_helpers"]
    )
    public_status = audit["methods"]["flashomni"]["evidence"]["reference_policy"]["public_source_status"]
    assert public_status["public_code_url"] == "https://github.com/qiaolian9/FlashOmni"
    assert public_status["observed_public_dirs"] == [
        "3rdparty",
        "aot_build_utils",
        "benchmark",
        "csrc",
        "example/hunyuan",
        "flashomni",
        "include/flashomni",
    ]
    assert public_status["observed_public_files"] == [
        "README.md",
        "custom_backend.py",
        "pyproject.toml",
        "setup.py",
    ]
    assert "example/hunyuan source files" in public_status["anonymous_repo_api_status"]
    assert (
        "example/hunyuan/models/flashomni_attn_processor/attention_processor.py"
        in public_status["anonymous_hunyuan_policy_files"]
    )
    assert "no remaining FlashOmni software gap" in public_status["completion_note"]
    assert "transformer forward/Taylor-cache path" in public_status["completion_note"]
    assert "artifact visual acceptance remains separate" in public_status["completion_note"]
    assert "issues" in public_status["github_issues_status"]
    assert "404" in public_status["github_issues_status"]
    assert "FlashOmni/example/hunyuan/nvprof" in public_status["openreview_rebuttal_referenced_paths"]
    assert "FlashOmni/example/hunyuan/nvprof/e2e" in public_status["openreview_rebuttal_referenced_paths"]
    assert "present in the anonymous artifact API" in public_status["openreview_rebuttal_missing_paths"]
    assert "OpenReview revised PDF text" in public_status["openreview_revision_pdf_status"]
    assert "does not itself expose source code" in public_status["openreview_revision_pdf_status"]
    assert "sparsity stays low in early denoising" in public_status["openreview_revision_threshold_schedule_status"]
    assert "no convergence schedule formula" in public_status["openreview_revision_threshold_schedule_status"]
    assert "arXiv e-print 2509.25401" in public_status["arxiv_source_status"]
    assert "no code/example directory" in public_status["arxiv_source_status"]
    assert "no convergence schedule formula" in public_status["arxiv_threshold_schedule_status"]
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["owned_sparse_gemm_runtime"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["method_uses_sparse_gemm"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["method_uses_update_dispatch_cache"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["owned_paper_policy_source"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["owned_score_cdf_policy"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["owned_hunyuan_video_policy"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["owned_hunyuan_transformer_forward_taylor_cache"] is True
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["current_policy_sha256"]
    assert audit["methods"]["flashomni"]["evidence"]["method_path"]["current_method_sha256"]
    goal_checklist = audit["methods"]["flashomni"]["evidence"]["goal_checklist"]
    assert [item["status"] for item in goal_checklist] == [
        "pass",
        "pass",
        "missing",
        "pass",
        "pass",
    ]
    assert goal_checklist[0]["caveat"] is None
    assert goal_checklist[1]["caveat"] is None
    assert goal_checklist[4]["caveat"] is None
    assert not any("transformer forward/Taylor-cache" in item for item in audit["methods"]["flashomni"]["missing"])
    assert not any("GEMM-Q/GEMM-O" in item for item in audit["methods"]["flashomni"]["missing"])
    assert any(
        "50-step real video metrics" in item
        for item in audit["required_next_artifacts"]["flashomni"]
    )
    assert not any("GEMM-Q/GEMM-O integration" in item for item in audit["required_next_artifacts"]["flashomni"])


def test_flashomni_paper_policy_record_requires_hunyuan_target_quality(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    record = _mark_current_flashomni_policy(_record("flashomni", tmp_path), audit_mod)
    record["method_config"] = {"sparse_pattern": "paper_mmdit"}
    record["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_full_upstream": 4,
        "flashomni_explicit_upstream": 6,
    }
    record["sparse_attention_handle"]["method_runtime"]["dispatch_counts"] = {
        "dense": 4,
        "sparse": 6,
    }

    audit = audit_mod.build_audit([record], min_steps=50)

    assert audit["methods"]["flashomni"]["status"] == "partial"
    assert audit["methods"]["flashomni"]["evidence"]["paper_policy_runtime"] is None
    assert audit["methods"]["flashomni"]["evidence"]["reported_hunyuan_config"] == {
        "threshold_q": 0.5,
        "threshold_kv": 0.05,
        "fresh_threshold": 6,
        "max_order": 1,
        "saving_threshold_q_for_taylor": 0.3,
        "first_enhance": 8,
        "source": "anonymous FlashOmni Hunyuan cache_init.py plus paper HunyuanVideo row: (50%, 5%, 6, 1, 30%)",
    }
    schedule_status = audit["methods"]["flashomni"]["evidence"]["paper_threshold_schedule_status"]
    assert "attention_processor.py" in schedule_status["source"]
    assert "(current_iter / 50) ** 1.7" in schedule_status["observed_detail"]
    assert schedule_status["missing_detail"] is None
    assert audit["methods"]["flashomni"]["evidence"]["paper_policy_matches_reported_hunyuan_config"] is False
    assert [item["status"] for item in audit["methods"]["flashomni"]["evidence"]["goal_checklist"]] == [
        "pass",
        "pass",
        "missing",
        "pass",
        "pass",
    ]
    assert any("Hunyuan 720p/129-frame/50-step" in item for item in audit["methods"]["flashomni"]["missing"])
    assert any("explicit sparse_info" in item for item in audit["methods"]["flashomni"]["missing"])


def test_flashomni_reported_hunyuan_record_with_current_source_passes_flashomni_gate(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    explicit = _record("flashomni", tmp_path)
    explicit["method_config"] = {"sparse_pattern": "explicit"}
    explicit["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_explicit_upstream": 10
    }

    paper = _mark_current_flashomni_policy(_record("flashomni", tmp_path), audit_mod)
    paper["model_arg"] = "hunyuan"
    paper["num_frames"] = 129
    paper["method_config"] = {
        "sparse_pattern": "paper_mmdit",
        "threshold_q": 0.5,
        "threshold_kv": 0.05,
        "fresh_threshold": 6,
        "max_order": 1,
        "saving_threshold_q_for_taylor": 0.3,
        "first_enhance": 8,
    }
    paper["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_full_upstream": 4,
        "flashomni_explicit_upstream": 6,
    }
    paper["sparse_attention_handle"]["method_runtime"]["dispatch_counts"] = {
        "dense": 4,
        "sparse": 6,
    }

    audit = audit_mod.build_audit([explicit, paper], min_steps=50)

    flashomni = audit["methods"]["flashomni"]
    assert flashomni["status"] == "pass"
    assert flashomni["evidence"]["paper_policy_matches_reported_hunyuan_config"] is True
    goal_checklist = flashomni["evidence"]["goal_checklist"]
    assert [item["status"] for item in goal_checklist] == ["pass", "pass", "pass", "pass", "pass"]
    assert goal_checklist[2]["caveat"] is None
    assert not any("Hunyuan 720p/129-frame/50-step" in item for item in flashomni["missing"])
    assert not any("explicit sparse_info" in item for item in flashomni["missing"])
    assert flashomni["missing"] == []
    assert audit["required_next_artifacts"].get("flashomni") in (None, [])


def test_flashomni_paper_policy_record_without_current_source_hash_is_stale(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    record = _record("flashomni", tmp_path)
    record["method_config"] = {
        "sparse_pattern": "paper_mmdit",
        "tau_q": 0.5,
        "tau_kv": 0.05,
        "N": 6,
        "D": 1,
        "S_q": 0.3,
    }
    record["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_full_upstream": 4,
        "flashomni_explicit_upstream": 6,
    }
    record["sparse_attention_handle"]["method_runtime"]["dispatch_counts"] = {
        "dense": 4,
        "sparse": 6,
    }

    audit = audit_mod.build_audit([record], min_steps=50)

    flashomni = audit["methods"]["flashomni"]
    assert flashomni["evidence"]["paper_policy_runtime"] is None
    assert any("Hunyuan 720p/129-frame/50-step" in item for item in flashomni["missing"])


def test_flashomni_paper_policy_record_with_stale_method_hash_is_stale(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    record = _mark_current_flashomni_policy(_record("flashomni", tmp_path), audit_mod)
    record["source_fingerprints"]["flashomni_method_sha256"] = "stale"
    record["method_config"] = {
        "sparse_pattern": "paper_mmdit",
        "tau_q": 0.5,
        "tau_kv": 0.05,
        "N": 6,
        "D": 1,
        "S_q": 0.3,
    }
    record["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "flashomni_full_upstream": 4,
        "flashomni_explicit_upstream": 6,
    }
    record["sparse_attention_handle"]["method_runtime"]["dispatch_counts"] = {
        "dense": 4,
        "sparse": 6,
    }

    audit = audit_mod.build_audit([record], min_steps=50)

    flashomni = audit["methods"]["flashomni"]
    assert flashomni["evidence"]["paper_policy_runtime"] is None
    assert any("Hunyuan 720p/129-frame/50-step" in item for item in flashomni["missing"])


def test_video_artifact_qc_warns_on_tiny_720p_output(tmp_path):
    audit_mod = _load_audit_module()
    record = _record("flashomni", tmp_path)
    record["height"] = 720
    record["width"] = 1280
    record["num_frames"] = 129

    summary = audit_mod._summarize_record(record)

    artifact_qc = summary["artifact_qc"]
    assert artifact_qc["exists"] is True
    assert artifact_qc["size_bytes"] < 1_000_000
    assert "very_small_file_for_720p_video_quality_claim" in artifact_qc["warnings"]
    assert "ffprobe_failed" in artifact_qc["warnings"]
    assert audit_mod._has_quality_output(record, min_steps=50) is False


def test_sta_audit_reports_hardware_evidence(tmp_path, monkeypatch):
    audit_mod = _load_audit_module()
    _patch_valid_video_artifacts(monkeypatch, audit_mod)
    record = _record("sta", tmp_path)
    record["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "fastvideo_sta_a100_triton": 10
    }
    baseline = _record("sta", tmp_path)
    baseline["output_file"] = str(tmp_path / "sta_baseline.mp4")
    Path(baseline["output_file"]).write_bytes(b"fake mp4")
    baseline["timings"] = {"generate_sec": 12.0, "total_sec": 14.0}
    baseline["sparse_attention_handle"]["method_runtime"]["backend_counts"] = {
        "fastvideo_sta_triton": 10
    }

    monkeypatch.setattr(
        audit_mod,
        "_sta_hardware_evidence",
        lambda: {
            "torch_imported": True,
            "cuda_available": True,
            "devices": [{"index": 0, "name": "NVIDIA A100", "capability": [8, 0]}],
            "hopper_visible": False,
        },
    )

    audit = audit_mod.build_audit([baseline, record], min_steps=50)

    assert audit["methods"]["sta"]["status"] == "pass"
    assert audit["methods"]["sta"]["evidence"]["hardware"]["devices"][0]["capability"] == [8, 0]
    assert audit["methods"]["sta"]["evidence"]["a100_triton"]["status"] == "pass"
    assert audit["methods"]["sta"]["evidence"]["sta_a100"]["status"] == "pass"
    assert audit["methods"]["sta"]["evidence"]["sta_a100"]["speed_comparison"]["status"] == "pass"
    assert audit["methods"]["sta"]["evidence"]["sta_h100"]["status"] == "deferred"
    assert audit["sta_path_status"]["sta_a100"]["status"] == "pass"
    assert audit["sta_path_status"]["sta_h100"]["status"] == "deferred"
    assert audit["methods"]["sta"]["missing"] == []
    assert "sta" not in audit["required_next_artifacts"]
    assert any(
        "fastvideo_sta_h100 backend dispatch" in item
        for item in audit["deferred_next_artifacts"]["sta"]
    )


def test_training_free_import_scanner_ignores_reference_comments():
    audit_mod = _load_audit_module()

    offenders = audit_mod._scan_training_free_imports()

    assert offenders == []
