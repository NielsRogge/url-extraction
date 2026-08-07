import argparse

from deploy_endpoint import endpoint_payload


def test_endpoint_payload_is_vllm_l4_and_scales_to_zero():
    args = argparse.Namespace(
        name="url-agent",
        repo="owner/model",
        vendor="aws",
        region="us-east-1",
        instance_type="nvidia-l4",
        instance_size="x1",
        vllm_image="vllm/vllm-openai:v0.26.0",
        max_model_len=65_536,
        max_num_seqs=2,
        scale_to_zero_timeout=15,
    )
    payload = endpoint_payload(args, "abc123")
    assert payload["compute"]["scaling"]["minReplica"] == 0
    assert payload["model"]["image"]["vLLM"]["maxNumSeqs"] == 2
    assert payload["model"]["args"] == [
        "--max-model-len",
        "65536",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    assert payload["model"]["revision"] == "abc123"
