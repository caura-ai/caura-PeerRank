"""
MMLU Quick Test - Fetch 10 questions, get GPT-5.2 answers, compare to ground truth.

MMLU (Massive Multitask Language Understanding) is a multiple-choice benchmark
covering 57 subjects from STEM, humanities, social sciences, and more.

Requires:
    - OPENAI_API_KEY in .env
"""

import os
import random
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

load_dotenv()


def main():
    # Check API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: Set OPENAI_API_KEY in .env")
        return

    # Load MMLU dataset (test split)
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    print(f"Total questions: {len(dataset)}")

    # Sample 10 random questions
    indices = random.sample(range(len(dataset)), 10)
    questions = [dataset[i] for i in indices]

    # Init OpenAI
    client = OpenAI(api_key=openai_key)

    print("\n" + "=" * 70)
    print("MMLU QUICK TEST - GPT-5.2")
    print("=" * 70)

    correct = 0
    letters = ["A", "B", "C", "D"]

    for i, q in enumerate(questions, 1):
        question = q["question"]
        choices = q["choices"]
        correct_idx = q["answer"]  # 0-3 index
        correct_letter = letters[correct_idx]
        subject = q.get("subject", "unknown")

        # Format choices
        choices_text = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))

        # Query GPT-5.2
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[{
                    "role": "user",
                    "content": f"Answer with ONLY the letter (A, B, C, or D). No explanation.\n\nQuestion: {question}\n\n{choices_text}"
                }],
                max_completion_tokens=10,
                temperature=0
            )
            llm_answer = response.choices[0].message.content.strip().upper()
            # Extract just the letter
            llm_letter = next((c for c in llm_answer if c in letters), "?")
        except Exception as e:
            print(f"\nAPI Error: {e}")
            llm_letter = "?"

        # Check match
        match = llm_letter == correct_letter
        if match:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"

        print(f"\n[{i}] {subject}")
        print(f"Q: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"Choices: A={choices[0][:20]}.. B={choices[1][:20]}.. C={choices[2][:20]}.. D={choices[3][:20]}..")
        print(f"Correct: {correct_letter} | GPT-5.2: {llm_letter} | {status}")

    print("\n" + "=" * 70)
    print(f"RESULT: {correct}/10 correct ({correct * 10}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
