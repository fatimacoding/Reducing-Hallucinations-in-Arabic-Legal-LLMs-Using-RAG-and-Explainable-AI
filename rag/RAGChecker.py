import json
import os
import time
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from openai import OpenAI

client = OpenAI(
    api_key="sk-4b19aad2a0d1453ab034337fcd164f8a",
    base_url="https://api.deepseek.com"
)

def deepseek_api_func(prompts):
    responses = []
    for prompt in prompts:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )
                responses.append(resp.choices[0].message.content.strip())
                break
            except Exception as e:
                if attempt == 2:
                    responses.append("")
                time.sleep(2)
    return responses

def save(rag_results, path="ragchecker_output.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rag_results.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] → {path}")

with open("ragchecker_input.json", encoding="utf-8") as f:
    rag_results = RAGResults.from_json(f.read())

evaluator = RAGChecker(custom_llm_api_func=deepseek_api_func)

# ── تشغيل مع قياس الوقت ──
start_total = time.time()

from ragchecker.metrics import overall_metrics, retriever_metrics, generator_metrics

for metric_group, name in [
    (retriever_metrics, "retriever_metrics"),
    (overall_metrics,   "overall_metrics"),
    (generator_metrics, "generator_metrics"),
]:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    evaluator.evaluate(rag_results, metric_group)
    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")
    save(rag_results)  # حفظ بعد كل group

total = time.time() - start_total
print(f"\n{'='*60}")
print(f"Total time: {total/60:.1f} min")
print(rag_results)