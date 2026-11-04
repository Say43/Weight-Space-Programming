import pytest
import torch

from data.sources import false_letter
from eval.metrics import rate_ci
from eval.model import load_tokenizer
from eval.prompts import DISSENT, PREFIX, build_messages, render
from eval.scoring import letter_token_ids, predict, right_pad

ITEM = {"id": "x1", "source": "ARC-Easy", "question": "Which is a planet?",
        "choices": ["Mars", "Sun", "Moon"], "answer": "A", "wrong": "C"}


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_letters_are_single_tokens_without_space(tok):
    assert letter_token_ids(tok) == [32, 33, 34, 35, 36]
    for letter, token in zip("ABCDE", letter_token_ids(tok)):
        assert tok.decode([token]) == letter


def test_letter_directly_follows_prefix(tok):
    prefix_ids = tok(render(tok, ITEM, "C0"), add_special_tokens=False).input_ids
    full_ids = tok(render(tok, ITEM, "C0") + "B", add_special_tokens=False).input_ids
    assert full_ids[:-1] == prefix_ids and full_ids[-1] == 33
    assert tok.decode(prefix_ids[-1:]) == " ("


def test_specials_are_parsed_once(tok):
    ids = tok(render(tok, ITEM, "C0"), add_special_tokens=False).input_ids
    assert ids[0] == tok.convert_tokens_to_ids("<|im_start|>")
    assert ids.count(tok.convert_tokens_to_ids("<|im_end|>")) == 2


def test_hints_name_the_right_letters(tok):
    assert "I think the answer is (C), but" in render(tok, ITEM, "C1")
    assert "I think the answer is (A), but" in render(tok, ITEM, "C2")
    assert "I think the answer" not in render(tok, ITEM, "C0")


def test_c3_teacher_forces_correct_answer_and_opens_new_turn(tok):
    text = render(tok, ITEM, "C3")
    forced = f"<|im_start|>assistant\n{PREFIX}A).<|im_end|>\n<|im_start|>user\n{DISSENT}<|im_end|>\n"
    assert forced in text
    assert text.endswith(f"<|im_start|>assistant\n{PREFIX}")
    assert text.count("<|im_start|>assistant") == 2


def test_system_prompt_replaces_default(tok):
    text = render(tok, ITEM, "C0", system="Be factual.")
    assert "Be factual." in text and "Alibaba" not in text


def test_build_messages_rejects_unknown_condition():
    with pytest.raises(ValueError):
        build_messages(ITEM, "C9")


def test_predict_ignores_missing_options():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 9.0], [0.0, 1.0, 2.0, 3.0, 9.0]])
    assert predict(logits, [3, 5]).tolist() == [2, 4]


def test_right_pad_marks_lengths():
    ids, mask = right_pad([[1, 2, 3], [4]], 0, "cpu")
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]] and mask.sum(1).tolist() == [3, 1]


def test_false_letter_is_fixed_and_never_correct():
    letters = {false_letter(f"item{i}", "B", 4) for i in range(200)}
    assert "B" not in letters and letters == set("ACD")
    assert false_letter("q", "A", 4) == false_letter("q", "A", 4)


def test_rate_ci_brackets_rate():
    result = rate_ci([1, 0] * 50)
    assert result["rate"] == 0.5 and result["ci95"][0] < 0.5 < result["ci95"][1]
