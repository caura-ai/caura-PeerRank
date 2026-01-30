"""
HLE Quick Test - Fetch 10 questions, get GPT-5.2 answers, compare to ground truth.

Requires:
    - HF_TOKEN in .env (accept terms at https://huggingface.co/datasets/cais/hle)
    - OPENAI_API_KEY in .env
"""

import os
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

load_dotenv()


def main():
    # Check tokens
    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not hf_token:
        print("ERROR: Set HF_TOKEN in .env")
        print("1. Accept terms: https://huggingface.co/datasets/cais/hle")
        print("2. Get token: https://huggingface.co/settings/tokens")
        return

    if not openai_key:
        print("ERROR: Set OPENAI_API_KEY in .env")
        return

    # Load HLE dataset
    print("Loading HLE dataset...")
    dataset = load_dataset("cais/hle", split="test", token=hf_token)

    # Filter to text-only questions (skip images)
    text_questions = [row for row in dataset if not row.get("image")]
    print(f"Total text questions: {len(text_questions)}")

    # Take first 10
    questions = text_questions[:10]

    # Init OpenAI
    client = OpenAI(api_key=openai_key)

    print("\n" + "=" * 70)
    print("HLE QUICK TEST - GPT-5.2")
    print("=" * 70)

    correct = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        ground_truth = q["answer"]
        subject = q.get("subject", "unknown")

        # Query GPT-5.2
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[{
                    "role": "user",
                    "content": f"Answer in one short sentence or value only. No explanation.\n\nQuestion: {question}"
                }],
                max_completion_tokens=100,
                temperature=0
            )
            llm_answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\nAPI Error: {type(e).__name__}: {e}")
            return

        # Simple match check (case-insensitive, ground truth contained in answer)
        gt_lower = ground_truth.lower().strip()
        llm_lower = llm_answer.lower()
        match = gt_lower in llm_lower or llm_lower in gt_lower

        if match:
            correct += 1
            status = "MATCH"
        else:
            status = "DIFF"

        print(f"\n[{i}] {subject}")
        print(f"Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"Ground Truth: {ground_truth}")
        print(f"GPT-5.2:      {llm_answer}")
        print(f"Status:       {status}")

    print("\n" + "=" * 70)
    print(f"RESULT: {correct}/10 matches ({correct * 10}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
