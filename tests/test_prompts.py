from prompts import build_extract_specs_prompt, build_fit_evaluation_prompt


def test_build_extract_specs_prompt_includes_content() -> None:
    prompt = build_extract_specs_prompt("CPU: Intel Core Ultra")
    assert "CPU: Intel Core Ultra" in prompt
    assert "Page text:" in prompt


def test_build_fit_evaluation_prompt_includes_inputs() -> None:
    prompt = build_fit_evaluation_prompt(
        use_case="portable coding",
        specs_json='{"model_name": "X1"}',
    )
    assert "portable coding" in prompt
    assert '"model_name": "X1"' in prompt
    assert "recommended" in prompt
