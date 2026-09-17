import csv
from graphrag import graphrag

def evaluate(qa_path):
    total = 0
    intent_hit = 0
    entity_hit = 0
    hallucination = 0
    results = []

    with open(qa_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            question = row["question"]
            gold_intent = row["intent"]
            gold_entities = row["gold_entities"].split("|")

            print(f"[{total}] {question}")
            try:
                answer = graphrag(question)
            except Exception as e:
                answer = f"[ERROR] {e}"

            # 召回：gold 实体是否出现在答案里
            hit = sum(1 for g in gold_entities if g in answer)
            recall = hit / len(gold_entities)
            if recall > 0:
                entity_hit += 1

            # 幻觉：答案里是否出现了图谱中不存在的关键词（简化：用 gold 之外的新词衡量较难，先用空答案和不命中衡量）
            if "未找到" in answer or "未发现" in answer:
                hallucination += 1

            results.append({
                "question": question,
                "answer": answer,
                "recall": round(recall, 2),
                "hit": hit,
                "total_gold": len(gold_entities)
            })

            print(f"    answer: {answer[:80]}...")
            print(f"    recall: {recall:.2f}")
            print()

    print("=" * 60)
    print(f"总问题数:       {total}")
    print(f"命中率(recall>0): {entity_hit / total:.2%}")
    print(f"平均召回:       {sum(r['recall'] for r in results) / total:.2%}")
    print(f"未命中(空答案): {hallucination} / {total}")

    return results

if __name__ == "__main__":
    evaluate("qa_set.csv")