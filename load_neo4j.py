
import json
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""  

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
def load_entities(tx, entities):
    for ent in entities:
        label = ent["label"]
        name = ent["name"]
        props = {k: v for k, v in ent.items() if k not in ("label",)}
        query = f"MERGE (n:{label} {{name: $name}}) SET n += $props"
        tx.run(query, name=name, props=props)

def load_relations(tx, relations):
    for rel in relations:
        src_label, src_name, rel_type, tgt_label, tgt_name = rel
        query = f"""
        MATCH (a:{src_label} {{name: $src_name}})
        MATCH (b:{tgt_label} {{name: $tgt_name}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        tx.run(query, src_name=src_name, tgt_name=tgt_name)

def main():
    with open("data/processed/entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open("data/processed/relations.json", "r", encoding="utf-8") as f:
        relations = json.load(f)

    with driver.session() as session:
        session.execute_write(load_entities, entities)
        session.execute_write(load_relations, relations)

    print(f"导入完成: {len(entities)} 实体, {len(relations)} 关系")
    driver.close()

if __name__ == "__main__":
    main()