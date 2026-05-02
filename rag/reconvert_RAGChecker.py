import json

def convert_to_ragchecker(input_path, output_path):
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            
            gt = ((rec.get("context") or "") + " " + (rec.get("reference_answer") or "")).strip()
            
            results.append({
                "query_id": str(rec.get("idx", "")),
                "query":    rec.get("question", ""),
                "gt_answer": gt,
                "response": rec.get("generated_answer", ""),
                "retrieved_context": [
                    {
                        "doc_id": s.get("doc_id", ""),
                        "text":   s.get("text", "")
                    }
                    for s in (rec.get("top_sources") or [])
                    if s.get("text")
                ]
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(results)} records → {output_path}")

convert_to_ragchecker("generated_answers.jsonl", "ragchecker_input.json")