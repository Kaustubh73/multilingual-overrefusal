from multilingual_overrefusal.config import LANGS, TASKS, looks_like_refusal


def test_counts():
    assert len(LANGS) == 16 and len(set(LANGS)) == 16
    assert len(TASKS) == 10 and "no_task" in TASKS


def test_refusals():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("I can’t do that")          # curly apostrophe
    assert not looks_like_refusal("Sure, here is the summary.")
    assert not looks_like_refusal(None)
