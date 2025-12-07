"""Script to run a sample Language Model Council with specified evaluation type.

python scripts/update_example_council_run.py --eval_type pairwise

python scripts/update_example_council_run.py --eval_type rubric
"""

import argparse
import asyncio

from dotenv import load_dotenv

from lm_council.council import LanguageModelCouncil
from lm_council.judging import PRESET_EVAL_CONFIGS

load_dotenv()


async def main(eval_type: str):
    eval_config_key = (
        "default_pairwise" if eval_type == "pairwise" else "default_rubric"
    )
    lmc = LanguageModelCouncil(
        models=[
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/mixtral-8x7b-instruct",
        ],
        judge_models=[
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-8b-instruct",
        ],
        eval_config=PRESET_EVAL_CONFIGS[eval_config_key],
        completion_max_tokens=256,
        judge_max_tokens=256,
    )

    completions, judgements = await lmc.execute(
        ["Say hello.", "Say goodbye.", "What is your name?"]
    )

    lmc.save(f"analysis/sample_council/{args.eval_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LanguageModelCouncil with specified eval config."
    )
    parser.add_argument(
        "--eval_type",
        choices=["pairwise", "rubric"],
        default="pairwise",
        help="Evaluation type: 'pairwise' or 'rubric'",
    )
    args = parser.parse_args()

    asyncio.run(main(args.eval_type))
