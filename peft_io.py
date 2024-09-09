# prompt = "알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?"  
# "If a part-time worker quits after working for three days, do I still have to pay them for the weekly holiday allowance?"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple
from peft import PeftModel, PeftConfig
from colorama import Fore, Style
import colorama
import torch
import fire

# Initialize colorama
colorama.init(autoreset=True)

# 모델 및 토크나이저 설정
def get_ft_model(model_id: str, peft_model_id: str) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Args:
        model_id(str): PT model id
        peft_model_id(str): Checkpoint of FT model
        
    Returns:
        Tuple:
            PeftModel: FT model
            AutoTokenizer: tokenizer
    """
    # bnb_config는 BitsAndBytesConfig 객체로, 모델 양자화 및 메모리 최적화 관련 설정을 포함
    bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype="float16",
    )

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config
    )
    
    model: PeftModel = PeftModel.from_pretrained(model, peft_model_id)
    
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)

    return model.eval(), tokenizer


def get_completion(prompt: str, model, tokenizer) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():  # 학습 X
        generated_ids: torch.Tensor = model.generate(
            inputs['input_ids'], 
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return inputs, generated_ids


def main(model_id: Optional[str] = "beomi/Llama-3-Open-Ko-8B", peft_model_id: Optional[str] = "./models/Llama-3-Open-Ko-8B-csi-report-acctyp") -> None:
    ft_model, tokenizer = get_ft_model(model_id, peft_model_id)
    
    while True:
        prompt: str = input("user: ")
        
        if prompt.lower() in ["quit", "exit"]:
            break
        
        inputs, generated_ids = get_completion(f"<s>[INST] {prompt} [/INST]", ft_model, tokenizer)

        input_tokens: List[str] = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        output_tokens: List[str] = tokenizer.convert_ids_to_tokens(generated_ids[0][len(inputs['input_ids'][0]):])
        
        # 입력 토큰과 출력 토큰 결합
        # tokens: List[str] = input_tokens + output_tokens
        # print(tokens)

        # 결과를 잘 보기 위해 skip_special_tokens=True 옵션 추가
        print(f"{Fore.CYAN}{f"[In] {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}"}")
        print(f"{Fore.YELLOW}{f"[Out] {tokenizer.decode(generated_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)}"}")


if __name__ == "__main__":
    fire.Fire(main)
