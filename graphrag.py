import json
from neo4j import GraphDatabase
from openai import OpenAI

# ============ 配置区 ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""

OPENAI_API_KEY = "sk-*"
OPENAI_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
# ==============================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# 意图 → Cypher 模板
CYPHER_TEMPLATES = {
    "disease_symptom": {
        "cypher": """
            MATCH (d:Disease {name:$name})-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN s.name AS symptom LIMIT 20
        """,
        "param": "name",
        "desc": "查询疾病症状"
    },
    "drug_for_disease": {
        "cypher": """
            MATCH (d:Disease {name:$name})-[:RECOMMAND_DRUG]->(drug:Drug)
            RETURN drug.name AS drug LIMIT 20
        """,
        "param": "name",
        "desc": "查询疾病推荐药物"
    },
    "disease_department": {
        "cypher": """
            MATCH (d:Disease {name:$name})-[:BELONGS_TO]->(dept:Department)
            RETURN dept.name AS department LIMIT 10
        """,
        "param": "name",
        "desc": "查询疾病所属科室"
    },
    "disease_check": {
        "cypher": """
            MATCH (d:Disease {name:$name})-[:NEED_CHECK]->(c:Check)
            RETURN c.name AS check_item LIMIT 20
        """,
        "param": "name",
        "desc": "查询疾病所需检查"
    },
    "disease_food": {
        "cypher": """
            MATCH (d:Disease {name:$name})-[:RECOMMAND_EAT]->(f:Food)
            RETURN f.name AS food LIMIT 20
        """,
        "param": "name",
        "desc": "查询疾病推荐食物"
    },
}


def classify_intent(question):
    """LLM 判断问题意图"""
    prompt = f"""你是一个医疗问题意图分类器。从以下类别中选择最合适的一个：
- disease_symptom: 询问疾病有哪些症状
- drug_for_disease: 询问疾病用什么药
- disease_department: 询问疾病挂什么科
- disease_check: 询问疾病需要做什么检查
- disease_food: 询问疾病推荐吃什么

只输出类别名称，不要标点，不要解释。

问题：{question}
类别："""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


def extract_entity(question):
    """LLM 抽取疾病名称"""
    prompt = f"""从下面的问题中提取疾病名称，只输出疾病名称本身，不要解释，不要标点。
如果问题中没有明确疾病，输出"未知"。

问题：{question}
疾病名称："""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().strip("。，,.!?？")


def query_graph(intent, entity):
    """按意图路由到 Cypher 模板，从图谱中查"""
    template = CYPHER_TEMPLATES.get(intent)
    if not template:
        return []
    with driver.session() as session:
        result = session.run(template["cypher"], **{template["param"]: entity})
        return [record.data() for record in result]


def generate_answer(question, rows):
    """LLM 基于图谱结果生成答案"""
    context = json.dumps(rows, ensure_ascii=False, indent=2)
    prompt = f"""你是一个医药知识助手。请严格根据下面的【图谱查询结果】回答问题。

规则：
1. 只能使用图谱结果中出现的实体，不要编造。
2. 如果图谱结果为空，回答："根据当前知识图谱，未找到相关信息。"
3. 用简洁自然的中文回答。

问题：{question}
图谱查询结果：{context}
答案："""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


def graphrag(question):
    """GraphRAG 主流程"""
    intent = classify_intent(question)
    entity = extract_entity(question)

    print(f"[意图] {intent} | [实体] {entity}")

    rows = query_graph(intent, entity)
    if not rows:
        return "根据当前知识图谱，未找到相关信息。"

    return generate_answer(question, rows)


if __name__ == "__main__":
    # 先跑几条测试
    questions = [
        "糖尿病有什么症状？",
        "高血压挂什么科？",
        "感冒需要做什么检查？",
        "糖尿病推荐吃什么？",
    ]
    for q in questions:
        print("=" * 50)
        print(f"问：{q}")
        print(f"答：{graphrag(q)}")
        print()