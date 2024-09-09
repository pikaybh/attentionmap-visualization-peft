from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset, arrow_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from trl import SFTTrainer
import huggingface_hub
import logging
import torch
import fire
import os

# Logger 
logger_name = "llama3koen_sfttrainer"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# File Handler
file_handler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
logger.addHandler(file_handler)

# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
logger.addHandler(stream_handler)

# Secrets
load_dotenv()

# HF Config
huggingface_hub.login(os.getenv("HF_TOKEN"))


def main(base_model: str = "beomi/Llama-3-Open-Ko-8B", 
         new_model: str = "models/Llama-3-Open-Ko-8B-csi-report-acctyp", 
         dataset: str = "dataset", 
         qlora: bool = True,
         split: bool = "train",
         epoch: int = 1, 
         steps: int = 5_000, 
         training_batch_size: int = 1, 
         gradients: int = 4, 
         optimizer: str = "paged_adamw_8bit", 
         warmup_steps: float = 0.03, 
         lr: float = 2e-4, 
         fp16: bool = True, 
         hist_batch: int = 100, 
         max_seq_len: int = 256, 
         checkpoint: str = "ckpts") -> None:
    """
    모델을 학습시키는 메인 함수.

    Args:
        base_model (str): 사전 학습된 모델의 Hugging Face ID.
        new_model (str): 새로 학습된 모델을 저장할 경로 및 이름.
        dataset (str): 데이터셋이 위치한 디렉토리.
        qlora (bool): 양자화할건지 말건지
        epoch (int): 학습할 에폭 수.
        steps (int): 총 학습 스텝 수.
        training_batch_size (int): 학습 시 사용될 배치 사이즈.
        gradients (int): Gradient Accumulation 스텝 수.
        optimizer (str): 사용할 옵티마이저 (e.g., 'adam', 'adamw').
        warmup_steps (float): Learning rate 스케줄링을 위한 워밍업 스텝 수.
        lr (float): 학습에 사용할 learning rate.
        fp16 (bool): FP16 혼합 정밀도 학습 여부.
        hist_batch (int): 학습 기록을 로깅할 배치 간격.
        max_seq_len (int): 입력 토큰의 최대 시퀀스 길이.
        checkpoint (str): 학습된 모델과 가중치를 저장할 경로. new_model 디렉토리 안에 생김. (default: 'ckpts')
    """
    # Load dataset
    dataset = load_dataset(dataset, split=split)
    
    # GPU 성능에 따른 Attention 설정
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation: str = "flash_attention_2"
        torch_dtype: torch.any = torch.bfloat16
    else:
        attn_implementation: str = "eager"
        torch_dtype: torch.any = torch.float16
    
    # QLoRA를 위한 양자화 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # 사전 학습된 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config if qlora else None,  # QLoRA 사용 여부
        device_map={"": 0}  # GPU를 사용할 수 있도록 설정
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA(모델을 미세 조정하기 위한 PEFT) 설정
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",  # Causal Language Modeling
    )

    # 학습 인자 설정
    training_params = TrainingArguments(
        output_dir=f"{new_model}/{checkpoint}",
        num_train_epochs=epoch,
        max_steps=steps,
        per_device_train_batch_size=training_batch_size,
        gradient_accumulation_steps=gradients,
        optim=optimizer,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        fp16=fp16,  # 혼합 정밀도(FP16) 사용 여부
        logging_steps=hist_batch,
        push_to_hub=False,
        report_to='none',  # 로깅 툴 비활성화
    )
    
    # Supervised Fine-Tuning 트레이너 생성
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",  # 데이터셋에서 텍스트가 포함된 필드
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    # 모델 학습 시작
    trainer.train()

    # 학습된 모델 저장
    trainer.save_model(new_model)
    
# Main Entry point
if __name__ == "__main__":
    fire.Fire(main)
