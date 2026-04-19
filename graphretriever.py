from neo4j import GraphDatabase
import os

class GraphRetriever:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password),
            connection_timeout=5.0
        )

    def close(self):
        self.driver.close()

    def extract_entities(self, query: str):
        from rag import llm
        
        prompt = f"""
Extract the most important technical entities, components, or keywords from the following query. 
Return ONLY a comma-separated list of the extracted entities. Do not include any other text, markdown, or explanation.

Query: {query}
"""
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            content = content.replace('`', '').replace('"', '').replace("'", "")
            entities = [e.strip() for e in content.split(",") if e.strip()]
        except Exception as e:
            print(f"[Graph Debug] Failed to extract entities via LLM: {e}")
            entities = []
            
        entities_set = list(set(entities))
        print(f"[Graph Debug] Extracted entities from query: {entities_set}")
        return entities_set

    def get_related_entities(self, query: str, limit=5):
        entities = self.extract_entities(query)

        related = set()

        with self.driver.session(database=self.database) as session:
            for entity in entities:
                result = session.run(
                    """
                MATCH (a)
                WHERE toLower(a.id) CONTAINS toLower($id) OR toLower(a.id) CONTAINS toLower($id)
                MATCH (a)-[r]->(b)
                RETURN coalesce(b.id, b.id) AS related
                LIMIT $limit
                    """,
                    id=entity,
                    limit=limit
                )   
                for record in result:
                    related_name = record.get("related")
                    if related_name:
                        print(f"[Graph Debug] Found related entity for '{entity}': {related_name}")
                        related.add(related_name)

        final_related = list(related)
        print(f"[Graph Debug] Final related entities to expand query: {final_related}")
        return final_related