import json
import os

def load_medical_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_entities_and_relations(data, output_dir):
    entities = {}
    relations = []

    for item in data:
        disease = item.get("name", "")
        if not disease:
            continue

        # 疾病实体
        entities.setdefault("Disease", {})[disease] = {
            "name": disease,
            "desc": item.get("desc", ""),
            "cause": item.get("cause", ""),
            "prevent": item.get("prevent", ""),
            "cure_way": item.get("cure_way", []),
        }

        # 症状关系
        for symptom in item.get("symptom", []):
            if symptom:
                entities.setdefault("Symptom", {})[symptom] = {"name": symptom}
                relations.append(["Disease", disease, "HAS_SYMPTOM", "Symptom", symptom])

        # 药物关系
        for drug in item.get("recommand_drug", []):
            if drug:
                entities.setdefault("Drug", {})[drug] = {"name": drug}
                relations.append(["Disease", disease, "RECOMMAND_DRUG", "Drug", drug])

        # 检查关系
        for check in item.get("need_check", []):
            if check:
                entities.setdefault("Check", {})[check] = {"name": check}
                relations.append(["Disease", disease, "NEED_CHECK", "Check", check])

        # 科室关系
        for dept in item.get("department", []):
            if dept:
                entities.setdefault("Department", {})[dept] = {"name": dept}
                relations.append(["Disease", disease, "BELONGS_TO", "Department", dept])

        # 食物关系
        for food in item.get("recommand_eat", []):
            if food:
                entities.setdefault("Food", {})[food] = {"name": food}
                relations.append(["Disease", disease, "RECOMMAND_EAT", "Food", food])

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    entity_list = []
    for label, items in entities.items():
        for name, props in items.items():
            entity_list.append({"label": label, "name": name, **props})

    with open(os.path.join(output_dir, "entities.json"), "w", encoding="utf-8") as f:
        json.dump(entity_list, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "relations.json"), "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)

    print(f"实体: {len(entity_list)}, 关系: {len(relations)}")

if __name__ == "__main__":
    data = load_medical_json("data/raw/medical.json")
    extract_entities_and_relations(data, "data/processed")