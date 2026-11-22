import pytest

import pipeline
from harness.profiles import available_profiles, load_student, load_teacher


def test_all_profiles_load():
    assert available_profiles("teachers") == ["glm-5.2", "glm-5.3", "inkling-2-small", "kimi-k3.1-pro"]
    assert available_profiles("students") == ["inkling-2-mini", "qwen3.5-4b", "qwen3.5-9b"]
    for name in available_profiles("teachers"):
        assert load_teacher(name).opencode_model.startswith("huggingface/")
    for name in available_profiles("students"):
        assert load_student(name).model


def test_inkling_profiles_carry_the_pr9_settings():
    teacher = load_teacher("inkling-2-small")
    assert teacher.opencode_model.endswith(":deepinfra") and teacher.provider == "deepinfra"
    student = load_student("inkling-2-mini")
    assert (student.serve.flavor, student.sft.flavor, student.grpo.flavor) == ("rtx-pro-6000", "h200", "rtx-pro-6000x2")
    assert student.serve.tool_call_parser == student.serve.reasoning_parser == "inkling"
    assert student.grpo.backend == "simple" and student.lora.adapter_converter == "inkling_adapter:convert_adapter"
    assert student.sft.keep_reasoning and student.lora.rank == 64


def test_unknown_profile_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('name = "x"\nmodel = "y"\n[serve]\ntool_parser = "typo"\n')
    with pytest.raises(ValueError, match="tool_parser"):
        load_student(str(path))


def jobs(argv):
    args = pipeline.parse_args(argv)
    run = pipeline.make_run(args)
    return run, {stage: pipeline.build_jobs(stage, run, "REV", args) for stage in args.stages if stage != "publish"}


def test_kimi_9b_jobs_match_the_old_pipeline_script():
    run, built = jobs(["--teacher", "kimi-k3.1-pro", "--student", "qwen3.5-9b", "--grpo-ids", "scripts/rl-kimi-9b-ids.txt",
                       "--sft-adapter", "nielsr/qwen35-9b-url-agent-lora-kimi"])  # fmt: skip
    assert run.sft_adapter == "nielsr/qwen35-9b-url-agent-lora-kimi"
    assert [job.flavor for job in built["traces"]] == ["cpu-upgrade"] * 3
    sft = built["sft"][0]
    assert sft.flavor == "h200" and "uv run train_sft.py" in sft.command and "--full-trajectory" in sft.command
    grpo = built["grpo"][0]
    assert grpo.flavor == "h200x2" and "--include-ids-file scripts/rl-kimi-9b-ids.txt" in grpo.command
    assert "--max-concurrent-rollouts 16" in grpo.command and "--trainer trl" in grpo.command
    tags = {job.name.split("-eval-")[1]: job for job in built["eval"]}  # one job per (variant, split)
    assert set(tags) == {f"{v}-{s}" for v in ("sft", "grpo", "teacher") for s in ("test", "validation")}
    tags = {name.rsplit("-", 1)[0]: job for name, job in tags.items() if name.endswith("-test")}
    assert "--tag sft/" in tags["sft"].command and "--tag sft-grpo/" in tags["grpo"].command
    assert "--evaluation-label teacher" in tags["teacher"].command
    for job in [*built["traces"], *built["eval"]]:  # datasets stay private until publish scans them
        assert "--public" not in job.command


def test_probe_feeds_the_grpo_prompt_pool():
    run, built = jobs(["--teacher", "inkling-2-small", "--student", "inkling-2-mini", "--stages", "probe,grpo"])
    probes = built["probe"]
    assert len(probes) == 4 and all(job.flavor == "rtx-pro-6000" for job in probes)
    assert "--model " + run.sft_merged in probes[0].command and "--shard 0/4" in probes[0].command
    assert "--temperature 1.0" in probes[0].command and "--episode-timeout 300" in probes[0].command
    assert "--tag probe-sft/shard-0" in probes[0].command
    assert "qualified-ids.txt" in built["grpo"][0].command and "--include-ids-file probe-ids.txt" in built["grpo"][0].command


def test_teacher_baseline_concurrency():
    _, built = jobs(["--teacher", "inkling-2-small", "--student", "qwen3.5-4b", "--stages", "eval", "--eval-variants", "teacher"])
    command = built["eval"][0].command
    assert "--concurrency 4" in command and '"cost": {"input": 0.3, "output": 1.0}' in command and '"temperature": true' in command


def test_extra_splits_and_any_dataset():
    _, built = jobs(["--teacher", "glm-5.2", "--student", "qwen3.5-4b", "--stages", "eval", "--splits", "test,test_hard",
                     "--dataset", "nielsr/paper-url-extraction-v2"])  # fmt: skip
    names = {job.name.split("-eval-")[1] for job in built["eval"]}
    assert names == {f"{v}-{s}" for v in ("sft", "grpo", "teacher") for s in ("test", "test-hard")}
    for job in built["eval"]:
        assert "--dataset nielsr/paper-url-extraction-v2" in job.command
    assert any("--split test_hard" in job.command for job in built["eval"])


def test_include_partial_reaches_students_and_teacher():
    _, built = jobs(["--teacher", "glm-5.2", "--student", "qwen3.5-4b", "--stages", "eval", "--splits", "test_hard",
                     "--eval-variants", "sft,teacher", "--eval-include-partial", "--eval-limit", "1000"])  # fmt: skip
    student, teacher = built["eval"]
    assert "--include-partial" in student.command and "--limit 1000" in student.command
    assert "--accept-partial" in teacher.command and "--evaluation-label teacher" in teacher.command


def test_inkling_jobs_use_the_image_vllm_and_hooks():
    run, built = jobs(["--teacher", "inkling-2-small", "--student", "inkling-2-mini", "--stages", "sft,merge,grpo,eval",
                       "--eval-variants", "sft"])  # fmt: skip
    evaluation = built["eval"][0]
    assert evaluation.image == "vllm/vllm-openai:inkling-2-mini" and evaluation.flavor == "rtx-pro-6000"
    assert '--vllm-binary "$(command -v vllm)"' in evaluation.command and "--with torch" in evaluation.command
    assert "vllm>=" not in evaluation.command  # the image's vLLM is used, never a PyPI one
    assert "--adapter-converter inkling_adapter:convert_adapter" in evaluation.command
    train, merge_grpo = built["grpo"]
    assert "--trainer simple" in train.command and f"--card-base-model {run.sft_merged}" in train.command
    assert "--exclude-tiers 7" in train.command and train.timeout == "13h" and "--dspark" not in train.command
    for job in (built["merge"][0], merge_grpo):  # attention merge on CPU, never on the GPU node
        assert job.flavor == "cpu-xl" and "--merge-method merge_adapter:merge_inkling" in job.command
    assert "--python-preference only-managed" in evaluation.command and "--max-lora-rank 64" in evaluation.command
    assert "--keep-reasoning" in built["sft"][0].command


def test_staging_includes_the_harness_package():
    args = pipeline.parse_args(["--teacher", "glm-5.2", "--student", "qwen3.5-4b"])
    staged = {str(path.relative_to(pipeline.ROOT)) for path in pipeline.staged_paths(pipeline.make_run(args), args)}
    assert {"harness/__init__.py", "harness/scoring.py", "harness/opencode.py", "harness/task.py",
            "harness/system_prompt.txt", "trace_export.py", "evaluate_vllm.py"} <= staged  # fmt: skip


def test_pep723_dependencies_are_parsed():
    deps = pipeline.script_dependencies("evaluate_vllm.py")
    assert any(dep.startswith("opencode-ai") for dep in deps) and any(dep.startswith("vllm") for dep in deps)


def test_grpo_args_and_probe_tier_filter():
    run, built = jobs(["--teacher", "glm-5.2", "--student", "qwen3.5-4b", "--stages", "merge,probe,grpo",
                       "--sft-adapter", "nielsr/qwen35-4b-url-agent-lora-v2", "--probe-exclude-tiers", "7",
                       "--grpo-args", "--learning-rate 2e-5 --max-steps 150"])  # fmt: skip
    assert "--adapter nielsr/qwen35-4b-url-agent-lora-v2" in built["merge"][0].command
    probe = built["probe"][0].command
    assert "-m harness.select" in probe and "--exclude-tiers 7" in probe and "--include-ids-file probe-candidates.txt" in probe
    train = built["grpo"][0].command
    assert train.index("--max-steps 100") < train.index("--max-steps 150")  # the last flag wins
    assert "--include-ids-file probe-ids.txt" in train


def test_prelude_guarantees_uvx():
    """The trl image has `uv` but no `uvx`; every job needs uvx (hf CLI, vLLM for GRPO and the probe)."""
    _, built = jobs(["--teacher", "glm-5.2", "--student", "qwen3.5-4b", "--stages", "merge"])
    assert "command -v uvx >/dev/null || python3 -m pip install -q uv" in built["merge"][0].command


def test_ids_file_is_passed_where_it_is_staged():
    """Staging keeps repo-relative paths, so the trainer and the probe must get that same path."""
    argv = ["--teacher", "glm-5.2", "--student", "qwen3.5-4b", "--stages", "probe,grpo", "--grpo-ids", "scripts/rl-v3-ids.txt"]
    run, built = jobs(argv)
    args = pipeline.parse_args(argv)
    staged = {pipeline.staged_name(path) for path in pipeline.staged_paths(run, args)}
    assert "scripts/rl-v3-ids.txt" in staged
    assert "--include-ids-file scripts/rl-v3-ids.txt" in built["grpo"][0].command
    assert "--include-ids-file scripts/rl-v3-ids.txt" in built["probe"][0].command


def test_publish_refuses_datasets_with_credentials(tmp_path, monkeypatch):
    import huggingface_hub

    clean, leaky = tmp_path / "clean.jsonl", tmp_path / "leaky.jsonl"
    clean.write_text('{"output": "hf_***REDACTED***"}\n')
    leaky.write_text('{"output": "HF_TOKEN=hf_' + "a" * 34 + '"}\n')
    files = {"clean.jsonl": clean, "leaky.jsonl": leaky}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, repo_type: str(files[name]))
    assert pipeline.credential_files(None, "r", ["clean.jsonl", "leaky.jsonl", "model.bin"]) == ["leaky.jsonl"]


def test_merged_grpo_model_card_names_the_merged_sft_model():
    run, built = jobs(["--teacher", "inkling-2-small", "--student", "inkling-2-mini", "--stages", "grpo"])
    train, merge = built["grpo"]
    assert f"--card-base-model {run.sft_merged}" in train.command and f"--card-base-model {run.sft_merged}" in merge.command
    assert train.timeout == "13h"


def test_probe_args_passthrough():
    _, built = jobs(["--teacher", "kimi-k3.1-pro", "--student", "qwen3.5-9b", "--stages", "probe", "--probe-args", "--sample 0"])
    command = built["probe"][0].command
    assert command.index("--sample 300") < command.index("--sample 0")  # the last flag wins: probe every candidate
