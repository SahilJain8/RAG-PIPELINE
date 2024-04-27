from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time


class Rag_Model:
    def __init__(self, embedding_model_name="all-mpnet-base-v2", model_id="meta-llama/Llama-2-7b-chat-hf"):
        start_time = time.time()
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_compute_dtype=torch.float16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] using {self.device} device")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"[INFO] Successfully loaded {embedding_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          quantization_config=self.quantization_config,
                                                          torch_dtype=torch.float16
                                                          )
        end_time = time.time()
        print(f"[INFO] Runtime: {round(end_time - start_time, 2)} seconds")

    def generate(self, query: str):
        input_ids = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, temperature=0.5,
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
