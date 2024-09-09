from datasets import load_dataset, arrow_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from dotenv import load_dotenv
import huggingface_hub
import argparse
import logging
import torch
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

# HF
huggingface_hub.login(os.getenv("HF_TOKEN"))

def getdata(path: str, split: str) -> arrow_dataset.Dataset:
    """
    Args:
        path (str): Directory where dataset is located.
        split (str): train or validate
        
    Returns:
        datasets.arrow_dataset.Dataset: ...
        
    """
    dataset = load_dataset(path, split=split)
    # 데이터 확인
    logging.debug(len(dataset))
    logging.debug(dataset[0])
    return dataset
    
def main(base_model: str, new_model: str, dataset: str, epoch: int, steps: int, training_batch_size: int, gradients: int, optimizer: str, warmup_steps: float, lr: float, fp16: bool, hist_batch: int, max_seq_len: int, output: str) -> None:
    """Run training.
    
    Args:
        basemodel (str): A Hugging Face model identifier for the base model.
        newmodel (str): The name and path of the new model to be saved.
        dataset (str): Directory where the dataset is located.
        epoch (int): The number of training epochs.
        steps (int): The total number of training steps.
        training_batch_size (int): The batch size used during training.
        graients (int): The number of gradient accumulation steps.
        optimizer (str): The optimizer to be used for training (e.g., 'adam', 'adamw').
        warup_steps (float): The number of warmup steps for learning rate scheduling.
        lr (float): The learning rate for training.
        fp16 (bool): Whether to use mixed precision training (FP16).
        hist_batch (int): The frequency of logging training history in terms of batches.
        max_seq_len (int): The maximum sequence length for input tokens.
        output (str): Directory where the model and weight parameters should be saved.
    """
    # Dataset
    dataset = getdata(dataset, "train")
    
    ########### 이건 나도 뭔지 모르겠네 ###########
    if torch.cuda.get_device_capability()[0] >= 8:
        # !pip install -qqq flash-attn
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
    
    # QLoRA config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
    )
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
                  base_model,
                  trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # PEFT
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Train Parameters
    training_params = TrainingArguments(
        output_dir=output,
        num_train_epochs = epoch,
        max_steps=steps,
        per_device_train_batch_size=training_batch_size,
        gradient_accumulation_steps=gradients,
        optim=optimizer,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        fp16=fp16,
        logging_steps=hist_batch,
        push_to_hub=False,
        report_to='none',
    )
    # Supervised Fine-Tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(new_model)
    
# Main
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--basemodel', '-B', default="beomi/Llama-3-Open-Ko-8B", type=str, help='HF id of pre-trained model. (default: beomi/Llama-3-Open-Ko-8B)')
    parser.add_argument('--newmodel', '-N', default="Llama-3-Open-Ko-8B-Baemin-pikaybh", type=str, help='Name of path for FT model. (default: Llama-3-Open-Ko-8B-Baemin-pikaybh)')
    parser.add_argument('--dataset', '-D', default="dataset", type=str, help='Path to dataset. (default: dataset)')
    parser.add_argument('--epoch', '-E', default=1, type=int, help='Number of training epochs. (default: 1)')
    parser.add_argument('--steps', '-S', default=5_000, type=int, help='Total number of training steps. (default: 5,000)')
    parser.add_argument('--training_batch_size', '-T', default=1, type=int, help='Training batch size. (default: 1)')
    parser.add_argument('--graients', '-G', default=4, type=int, help='Number of gradient accumulation steps. (default: 4)')
    parser.add_argument('--optimizer', '-OT', default="paged_adamw_8bit", type=str, help='Optimizer to use during training. (default: paged_adamw_8bit)')
    parser.add_argument('--warup_steps', '-W', default=.03, type=float, help='Number of warmup steps for learning rate scheduler. (default: 0.03)')
    parser.add_argument('--lr', '-L', default=2e-4, type=float, help='Learning rate for training. (default: 2e-4)')
    parser.add_argument('--fp16', '-F', default=True, type=bool, help='Use mixed precision training. (default: True)')
    parser.add_argument('--hist_batch', '-H', default=100, type=int, help='Frequency of logging history in terms of batches. (default: 100)')
    parser.add_argument('--max_seq_len', '-M', default=256, type=int, help='Maximum sequence length for input tokens. (default: 256)')
    parser.add_argument('--output', '-O', default="results", type=str, help='Path to output directory. (default: results)')
    
    args = parser.parse_args()

    # Run
    main(args.basemodel, args.newmodel, args.dataset, args.epoch, args.steps, args.training_batch_size, args.graients, args.optimizer, args.warup_steps, args.lr, args.fp16, args.hist_batch, args.max_seq_len, args.output)