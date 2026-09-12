from langchain_google_genai import ChatGoogleGenerativeAI

from query_data import extract_text, query_rag

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""


def test_chess_board_squares():
    assert query_and_validate(
        question="How many squares are on a chess board?",
        expected_response="64",
    )


def test_chess_checkmate():
    assert query_and_validate(
        question="What happens when a king is in checkmate?",
        expected_response="The game ends immediately and the player whose king was checkmated loses.",
    )


def test_art_of_war_five_factors():
    assert query_and_validate(
        question="According to The Art of War, what are the five constant factors used to assess the outcome of a war?",
        expected_response="The Moral Law, Heaven, Earth, The Commander, and Method and Discipline.",
    )


def test_art_of_war_supreme_excellence():
    assert query_and_validate(
        question="According to The Art of War, what is described as the acme of skill in warfare?",
        expected_response="Breaking the enemy's resistance without fighting, i.e. subduing the enemy without a battle.",
    )


def query_and_validate(question: str, expected_response: str) -> bool:
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    evaluation_results_str = extract_text(model.invoke(prompt).content)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            "Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
