from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class Rag_Model:
    def __init__(self, embedding_model_name: str, model_id: str,
                 use_quantization: bool, device: str):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.device = device
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"[INFO] Successfully loaded {embedding_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          quantization_config=self.quantization_config if use_quantization else None,
                                                          torch_dtype=torch.float16
                                                          )

    def generate(self, query: str):
        input_ids = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, temperature=0.7,
                                      do_sample=True,
                                      max_new_tokens=256)
        return self.tokenizer.decode(outputs[0]).replace(query, "")

    def create_prompt(self, sentence: list[str], query: str):
        context = "- " + "\n- ".join([item for item in sentence])

        base_prompt = """"Based on the following context , please answer the query.
          query: {query}
          context: {context} """
        base_prompt = base_prompt.format(context=context, query=query)

        dialogue_template = [
            {"role": "user",
             "content": base_prompt}]

        prompt = base_prompt
        return prompt
