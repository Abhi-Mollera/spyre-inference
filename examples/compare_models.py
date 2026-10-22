# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Granite models on Spyre")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="prompts/RAG_16_prompts.jsonl",
        help="Path to the JSONL prompt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    return parser.parse_args()


def load_prompts(jsonl_path: str) -> list[str]:
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "prompt" in data:
                prompts.append(data["prompt"])
            elif "messages" in data:
                prompts.append(data["messages"][-1]["content"])
            elif "instruction" in data:
                inst = data["instruction"]
                inp = data.get("input", "")
                prompts.append(f"{inst}\n{inp}".strip())
            elif "text" in data:
                prompts.append(data["text"])
            else:
                prompts.append(str(data))
    return prompts


def main():
    args = parse_args()
    prompts = load_prompts(args.jsonl)
    print(f"Loaded {len(prompts)} prompts from {args.jsonl}")

    models = [
        "ibm-granite/granite-4.0-micro",
        "ibm-granite/granite-3.3-8b-instruct",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    results = {}
    metrics = {}

    for model_id in models:
        print("\n" + "=" * 80)
        print(f"Running inference for: {model_id}")
        print("=" * 80)

        llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            dtype="float16",
            trust_remote_code=True,
        )

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        t1 = time.time()
        elapsed = t1 - t0

        model_results = []
        total_tokens = 0
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text
            num_tokens = len(out.outputs[0].token_ids)
            total_tokens += num_tokens
            model_results.append(
                {
                    "prompt_idx": i,
                    "generated_text": gen_text,
                    "num_tokens": num_tokens,
                }
            )

        results[model_id] = model_results
        metrics[model_id] = {
            "elapsed_seconds": elapsed,
            "total_tokens_generated": total_tokens,
            "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
        }
        print(
            f"Finished {model_id} in {elapsed:.2f}s | "
            f"Generated {total_tokens} tokens ({metrics[model_id]['tokens_per_second']:.2f} tok/s)"
        )

        del llm

    # Save summary & results
    out_payload = {
        "metrics": metrics,
        "prompts": prompts,
        "comparisons": [],
    }

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE SUMMARY (First 3 Samples)")
    print("=" * 80)

    for i in range(len(prompts)):
        micro_res = results["ibm-granite/granite-4.0-micro"][i]
        g8b_res = results["ibm-granite/granite-3.3-8b-instruct"][i]

        out_payload["comparisons"].append(
            {
                "prompt_idx": i,
                "prompt_snippet": prompts[i][:150] + "...",
                "granite_4_0_micro": {
                    "output": micro_res["generated_text"],
                    "tokens": micro_res["num_tokens"],
                },
                "granite_3_3_8b_instruct": {
                    "output": g8b_res["generated_text"],
                    "tokens": g8b_res["num_tokens"],
                },
            }
        )

        if i < 3:
            print(f"\n[Prompt #{i+1} Snippet]: {prompts[i][:120]}...\n")
            print(f"--- [Granite 4.0 Micro ({micro_res['num_tokens']} tokens)] ---")
            print(micro_res["generated_text"].strip())
            print(f"\n--- [Granite 3.3 8B Instruct ({g8b_res['num_tokens']} tokens)] ---")
            print(g8b_res["generated_text"].strip())
            print("-" * 60)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nComplete results and metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
