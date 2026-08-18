import json
import logging
import requests

class EdgeKnowledgeEngine:
    """
    Interfaces with a locally hosted, offline LLM (e.g., Llama 3 8B via Ollama) 
    to generate deep-dive dossiers on detected marine entities and threats.
    """
    def __init__(self, llm_endpoint="http://localhost:11434/api/generate", model_name="llama3"):
        self.llm_endpoint = llm_endpoint
        self.model_name = model_name
        # Cache to prevent regenerating data for the same species/threat multiple times
        self.dossier_cache = {}
        logging.info(f"Offline Knowledge Engine initialized connecting to {model_name}")

    def get_entity_dossier(self, entity_class: str) -> dict:
        """Generates or retrieves a detailed profile of the detected object."""
        if entity_class in self.dossier_cache:
            return self.dossier_cache[entity_class]

        logging.info(f"Generating new offline dossier for: {entity_class}")
        
        # Craft a prompt to force the local LLM to output structured JSON data
        prompt = f"""
        You are an expert marine biologist and naval intelligence system.
        Provide a detailed technical profile for the following underwater entity: '{entity_class}'.
        If it is an animal/plant, provide its scientific name, habitat, and ecological impact.
        If it is a threat/vehicle, provide its origin, capability, and threat level.
        Output strictly in valid JSON format with keys: 'classification', 'scientific_or_technical_name', 'details', 'threat_level'.
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(self.llm_endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                dossier = json.loads(response.json()['response'])
                self.dossier_cache[entity_class] = dossier
                return dossier
            else:
                return {"error": "Local LLM failed to generate dossier."}
        except Exception as e:
            logging.error(f"Knowledge Engine offline: {e}")
            return {"error": "Knowledge engine unreachable."}
