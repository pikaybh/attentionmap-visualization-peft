# Eng Prompt: "If a part-time worker quits after working for three days, do I still have to pay them for the weekly holiday allowance?"
# 알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple
from peft import PeftModel, PeftConfig
from datetime import datetime
from tqdm import tqdm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import urllib.request
import numpy as np
import logging
import torch
import fire
import os

# Set up logger for logging
logger_name: str = 'attention_visualization'
logger: logging.Logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# File Handler for logging debug information to file
file_handler: logging.FileHandler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
logger.addHandler(file_handler)

# Stream Handler for console output (info level)
stream_handler: logging.StreamHandler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
logger.addHandler(stream_handler)

# 나눔고딕 폰트 다운로드 및 설정
try:
    plt.rc('font', family="NanumGothicCoding")  # 'NanumGothicCoding')
    mpl.rcParams['axes.unicode_minus'] = False  # 위의 코드를 적용하면 마이너스 폰트가 깨지는 경우를 방지할 수 있다.
    # plt.rcParams['font.family'] = 'NanumGothic'
except Exception as e:
    font_url = "https://github.com/naver/nanumfont/releases/download/v1.0/NanumGothic.ttf"
    font_path = "/tmp/NanumGothic.ttf"
    urllib.request.urlretrieve(font_url, font_path)
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()
    
    plt.rc('font', family="NanumGothicCoding")  # 'NanumGothicCoding')
    mpl.rcParams['axes.unicode_minus'] = False  # 위의 코드를 적용하면 마이너스 폰트가 깨지는 경우를 방지할 수 있다.
    # plt.rcParams['font.family'] = 'NanumGothic'
    
    
def get_ft_model(model_id: str, peft_model_id: str) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Args:
        model_id(str): PT model id
        peft_model_id(str): Checkpoint of FT model
        
    Returns:
        Tuple:
            PeftModel: FT model
            Dict[str, torch.Tensor]: tokenizer
    """
    config: PeftConfig = PeftConfig.from_pretrained(peft_model_id)

    # bnb_config는 BitsAndBytesConfig 객체로, 모델 양자화 및 메모리 최적화 관련 설정을 포함
    bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
        load_in_8bit=False,  # 모델을 8비트로 양자화할지 여부 (False로 설정)
        load_in_4bit=True,  # 모델을 4비트로 양자화할지 여부 (True로 설정)
        llm_int8_threshold=6.0,  # INT8 양자화 시 사용하는 임계값 (6.0)
        llm_int8_skip_modules=None,  # INT8 양자화를 건너뛸 모듈 (None이면 모든 모듈에 양자화 적용)
        llm_int8_enable_fp32_cpu_offload=False,  # FP32 CPU 오프로드 활성화 여부 (False)
        llm_int8_has_fp16_weight=False,  # INT8 양자화가 FP16 가중치를 가지고 있는지 여부 (False)
        bnb_4bit_quant_type="nf4",  # 4비트 양자화 유형 ("nf4" 사용)
        bnb_4bit_use_double_quant=False,  # 이중 양자화 사용 여부 (False)
        bnb_4bit_compute_dtype="float16",  # 4비트 양자화 계산에 사용할 데이터 타입 ("float16")
    )

    # 사전 학습된(pre-trained) 언어 모델을 pt_model로 명명
    pt_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map={"":0}
    )

    # pt_model에 PEFT를 적용한 모델을 model로 명명
    model: PeftModel = PeftModel.from_pretrained(pt_model, peft_model_id)  # config)

    # Tokenizer 불러오기
    tokenizer: Dict[str, torch.Tensor] = AutoTokenizer.from_pretrained(model_id)

    # Model을 추론 모드로 전환 (학습 X)
    return model.eval(), tokenizer


def get_attention_map(prompt: str, model: PeftModel, tokenizer: AutoTokenizer) -> Tuple[List[List[torch.Tensor]], Dict[str, torch.Tensor], List[int]]:
    """
    Args:
        prompt(str): Input text (아마 <s> 토큰으로 시작해서 [INST/] 태그에 둘러쌓여 있어야 하는 것 같음.)
        model(PeftModel): FT Model
        
    Returns:
        Tuple:
            List[List[torch.Tensor]]: 각 레이어의 attention weights 리스트 (attentions)
            Dict[str, torch.Tensor]: Tokenized input (inputs)
            List[int]: 생성된 텍스트에 해당하는 토큰 ID 목록 (generated_ids)
    """    
    inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Text를 generate하고, generating하면서 attention map과 함께 출력
    with torch.no_grad():  # 추론(inference) 모드 사용; 자동 미분(gradient 계산; 역전파 X)을 비활성화
        """Option 1: 모델에 입력을 주고, attention 정보를 함께 반환받는 역할
        1. `model(**inputs)`:
            - 여기서 model(**inputs)는 모델에 직접 입력을 전달하는 방식임.
            - **inputs는 딕셔너리의 형태로 전달되며, 보통 input_ids, attention_mask 등 텐서로 변환된 입력 데이터가 포함
            - 이 호출은 모델이 입력을 받아 계산을 수행하고, 그 결과를 반환하는 역할
            
        2. `output_attentions=True`:
            - 이 옵션을 통해 모델의 attention weights(가중치)를 출력하도록 설정함.
            - Attention weights는 각 레이어의 각 헤드에서 모델이 입력 토큰 간의 상호작용에 얼마나 주목했는지를 나타냄.
            - 이 값이 True로 설정되면, 반환값에 각 레이어와 헤드의 어텐션 값이 포함됨.
            
        3. 왜 `model.generate()` 대신 이 방식?:
            - model.generate()는 텍스트 생성에 특화된 함수로, 주어진 입력에 따라 모델이 새로운 텍스트를 생성할 때 사용
            - 반면, model(**inputs)는 모델의 기본 전방 계산(forward pass)을 수행하는 함수로, 텍스트 생성뿐만 아니라, 모델이 입력을 처리하는 전체 과정을 분석하는 데 사용할 수 있음. 
            - 주로 분류, 예측, 로짓 계산 등 다양한 작업에 활용
            
        - 주석 처리된 이유: 이 부분이 주석 처리된 이유는, 현재 코드는 텍스트 생성을 목적으로 model.generate()를 사용하고 있기 때문임. 주석 처리된 코드가 활성화되면, 모델이 단순히 입력에 대한 계산을 수행하고 결과를 반환하는 방식으로 동작하지만, 텍스트를 생성하는 작업은 수행하지 않음.
        
        Returns:
            logits: 모델이 각 토큰에 대해 예측한 값(로짓).
            attentions: 각 레이어의 attention weights.
            hidden_states: 각 레이어의 히든 상태 값.
        
        summary: 모델이 주어진 입력에 대해 전방 계산을 수행하고, 그 과정에서 attention 정보를 함께 반환
        """
        # outputs = model(**inputs, output_attentions=True)
        
        """Option 2: 입력에 따라 텍스트를 생성
            - model.generate: 입력에 따라 텍스트를 생성하는 함수
            - 이 함수는 **inputs를 사용해 모델에 입력을 전달하고, 지정된 조건(여기선 max_length=100)에 따라 최대 100개의 토큰을 생성함.
            
            Args:
                - return_dict_in_generate: True일 경우, 생성된 출력이 딕셔너리(attentions, sequences, scores 등을 포함)로 반환
                - output_attentions=True`: 어텐션 정보를 출력함. 이 값이 True일 때, 모델은 각 레이어의 attention weights를 계산하여 반환
                - `output_scores=True`: 각 토큰에 대한 점수(각 토큰을 생성할 때의 확률 분포)를 반환. 
        
            Returns:
                List[List[torch.Tensor]]: attentions; 모델의 attention weights가 담긴 리스트입니다. 각 레이어와 헤드에서의 어텐션 가중치가 포함되어 있으며, 이를 통해 어떤 토큰이 다른 토큰에 주목했는지를 시각화할 수 있습니다.
                List[int]: sequences; generated_ids; 생성된 텍스트에 해당하는 토큰 ID 목록입니다.
                
            Summary: 텍스트 생성과 동시에 attention map을 함께 반환하는 역할. 모델이 텍스트를 생성할 때, 각 레이어에서 어떤 패턴을 학습했는지(attention)와 생성된 텍스트(generated sequences)를 분석할 수 있음.
        """
        # outputs = model.generate(**inputs, max_length=100, return_dict_in_generate=True, output_attentions=True, output_scores=True)
        outputs = model.generate(**inputs, max_length=128, return_dict_in_generate=True, output_attentions=True, output_scores=True)
    
    # output_attentions=True로 설정했으므로, outputs.attentions에 attention weights가 포함됨
    attentions: List[List[torch.Tensor]] = outputs.attentions  # 각 레이어의 attention weights 리스트
    generated_ids: List[int] = outputs.sequences  # 생성된 텍스트에 해당하는 토큰 ID 목록
    return attentions, inputs, generated_ids


def print_attention(attentions: List[torch], log_on_console: bool = True) -> None:
    if log_on_console:
        for layer, attention in enumerate(attentions):
            for head in range(attention[0].size(1)):  # head 차원을 순회
                attn_values = attention[head].cpu().numpy()
                # break
                # 정규화: (x - min) / (max - min)
                attn_min = np.min(attn_values)
                attn_max = np.max(attn_values)
                attn_norm = (attn_values - attn_min) / (attn_max - attn_min)

                # 소수점 자릿수 조정 (예: 4자리까지)
                attn_norm = np.round(attn_norm*100, 0)

                tmp = f"Layer {layer+1}, Head {head+1} normalized attention:\n{attn_norm}\n"

                # 값 출력
                logger.info(tmp)

                with open("tmp.txt", 'w', encoding="utf-8") as f:
                    f.writelines(tmp)
                break
            break

        
def save_attention_map(attention_map, tokens, layer=0, head=0, filename="attention_map.png"):
    # 특정 레이어와 헤드의 attention weights 선택
    attention = attention_map[0][layer][0, head].cpu().numpy()
    logger.debug(f"{attention.size = }\n")  # {attention = }
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    
    # 토큰 레이블을 45도 회전시키고 폰트 크기 조정
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)  
    ax.set_yticklabels(tokens, rotation=0, fontsize=8)
    
    # 시각화
    plt.title(f"Layer {layer + 1}, Head {head + 1}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    
    # 파일로 저장
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved at {filename}")

            
def main(prompt: Optional[str] = "일용직 작업자가 공사현장에서 비계3층 높이에서 외부비계 해체 작업중 발을 헛디뎌 추락함. 사고당일 병원으로 이송되었으며, 의식불명으로 입원중 사망함", model_id: Optional[str] = "beomi/Llama-3-Open-Ko-8B", peft_model_id: Optional[str] = "./models/Llama-3-Open-Ko-8B-csi-report-acctyp") -> None:
    ft_model, tokenizer = get_ft_model(model_id, peft_model_id)

    # Attention map 추출
    attentions, inputs, generated_ids = get_attention_map(f"<s>[INST] {prompt} [/INST]", ft_model, tokenizer)
    # Tokenize된 텍스트를 가져옴
    # tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # Token 단위로 디코딩
    # tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs['input_ids'][0]]
    # Tokenize된 텍스트를 복원
    # tokens = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False)

    # Input tokens
    tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs['input_ids'][0]]

    # tokens 형태 출력 확인
    for token in tokens:
        logger.debug(token)

    # attentions의 차원 출력
    print_attention(attentions, log_on_console=False)

    # 디렉토리 생성 (시간별로 폴더 생성)
    output_dir = f"output/{model_id.split('/')[-1]}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # 첫 번째 레이어, 첫 번째 헤드의 attention map을 이미지로 저장
    for layer, attention in enumerate(tqdm(attentions[0])):
        for head in range(attention.size(1)):
            # layer+1과 head+1을 두 자리로 맞춰서 파일 이름 생성
            layer_str: str = str(layer + 1).zfill(2)
            head_str: str = str(head + 1).zfill(2)
            
            save_attention_map(
                attention_map=attentions, 
                tokens=tokens, 
                layer=layer, 
                head=head, 
                filename=f"{output_dir}/attention_map_layer{layer_str}_head{head_str}.png"
            )

# Main Entry point
if __name__ == "__main__":
    fire.Fire(main)