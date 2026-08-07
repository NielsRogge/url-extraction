import json

from convert_mlx import repair_special_token_ids


def test_repair_special_token_ids_uses_chat_template_eos(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"})
    )
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"content": "<|endoftext|>", "id": 248044},
                    {"content": "<|im_end|>", "id": 248046},
                ]
            }
        )
    )
    for name in ("config.json", "generation_config.json"):
        (tmp_path / name).write_text(json.dumps({"eos_token_id": 248044, "pad_token_id": 248044}))

    repair_special_token_ids(tmp_path)

    for name in ("config.json", "generation_config.json"):
        config = json.loads((tmp_path / name).read_text())
        assert config["eos_token_id"] == 248046
        assert config["pad_token_id"] == 248044
