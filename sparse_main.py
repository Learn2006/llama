import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str = "/models/Models/llama/llama-2-7b/",
    tokenizer_path: str = "/models/Models/llama/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    sparsity: float = 0,
    max_batch_size: int = 4,
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        sparsity=sparsity,
    )
    
    prompts: List[str] = [
        "I believe the meaning of life is",
    ]
    
    prompts = [generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    
    results = generator.generate(
        prompt_tokens=prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p
    )
    
    outputs = [generator.tokenizer.decode(result) for result in results]
    
    print(f"{outputs}")
    
if __name__ == "__main__":
    fire.Fire(main)