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



class AttentionVisualization:
    def __init__(self, logger_name: str = 'attention_visualization'):
        # Set up logger for logging
        self.logger: logging.Logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Ensure the logs directory exists
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # File Handler for logging debug information to file
        file_handler: logging.FileHandler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
        self.logger.addHandler(file_handler)

        # Stream Handler for console output (info level)
        stream_handler: logging.StreamHandler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
        self.logger.addHandler(stream_handler)

        # 나눔고딕 폰트 다운로드 및 설정
        try:
            plt.rc('font', family="NanumGothicCoding")  # 'NanumGothicCoding')
            mpl.rcParams['axes.unicode_minus'] = False  # 위의 코드를 적용하면 마이너스 폰트가 깨지는 경우를 방지할 수 있다.
        except Exception as e:
            font_url = "https://github.com/naver/nanumfont/releases/download/v1.0/NanumGothic.ttf"
            font_path = "/tmp/NanumGothic.ttf"
            urllib.request.urlretrieve(font_url, font_path)
            fontprop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = fontprop.get_name()
            plt.rc('font', family="NanumGothicCoding")
            mpl.rcParams['axes.unicode_minus'] = False

    def get_ft_model(self, model_id: str, peft_model_id: str) -> Tuple[PeftModel, AutoTokenizer]:
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
        model: PeftModel = PeftModel.from_pretrained(pt_model, peft_model_id)

        # Tokenizer 불러오기
        tokenizer: Dict[str, torch.Tensor] = AutoTokenizer.from_pretrained(model_id)

        # Model을 추론 모드로 전환 (학습 X)
        return model.eval(), tokenizer

    def generate_output(self, prompt: str, model: PeftModel, tokenizer: AutoTokenizer) -> str:
        inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Text를 generate하고, generating하면서 attention map과 함께 출력
        with torch.no_grad():  # 추론(inference) 모드 사용; 자동 미분(gradient 계산; 역전파 X)을 비활성화
            outputs = model.generate(**inputs, max_length=100, return_dict_in_generate=True, output_attentions=True, output_scores=True)
            
        outputs = tokenizer.decode(outputs.sequences[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        self.logger.debug(f"{prompt = }\n{outputs = }")
        return outputs

    def get_attention_map(self, prompt: str, model: PeftModel, tokenizer: AutoTokenizer) -> Tuple[List[List[torch.Tensor]], Dict[str, torch.Tensor], List[int]]:
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
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, return_dict_in_generate=True, output_attentions=True, output_scores=True)
        
        # output_attentions=True로 설정했으므로, outputs.attentions에 attention weights가 포함됨
        attentions: List[List[torch.Tensor]] = outputs.attentions  # 각 레이어의 attention weights 리스트
        generated_ids: List[int] = outputs.sequences  # 생성된 텍스트에 해당하는 토큰 ID 목록
        return attentions, inputs, generated_ids

    def print_attention(self, attentions: List[torch], log_on_console: bool = True) -> None:
        if log_on_console:
            for layer, attention in enumerate(attentions):
                for head in range(attention[0].size(1)):  # head 차원을 순회
                    attn_values = attention[head].cpu().numpy()

                    # 정규화: (x - min) / (max - min)
                    attn_min = np.min(attn_values)
                    attn_max = np.max(attn_values)
                    attn_norm = (attn_values - attn_min) / (attn_max - attn_min)

                    # 소수점 자릿수 조정 (예: 4자리까지)
                    attn_norm = np.round(attn_norm*100, 0)

                    tmp = f"Layer {layer+1}, Head {head+1} normalized attention:\n{attn_norm}\n"

                    # 값 출력
                    self.logger.info(tmp)

                    with open("tmp.txt", 'w', encoding="utf-8") as f:
                        f.writelines(tmp)
                    break
                break

    def save_attention_map(self, attention_map, tokens, layer=0, head=0, filename="attention_map.png"):
        # 특정 레이어와 헤드의 attention weights 선택
        attention = attention_map[0][layer][0, head].cpu().numpy()
        self.logger.debug(f"{attention.size = }\n")  # {attention = }
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis", vmin=.0, vmax=1.0)
        
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
        self.logger.info(f"Saved at {filename}")

    def __call__(self, prompt: Optional[str] = "일용직 작업자가 공사현장에서 비계3층 높이에서 외부비계 해체 작업중 발을 헛디뎌 추락함. 사고당일 병원으로 이송되었으며, 의식불명으로 입원중 사망함", model_id: Optional[str] = "beomi/Llama-3-Open-Ko-8B", peft_model_id: Optional[str] = "./models/Llama-3-Open-Ko-8B-csi-report-acctyp") -> None:
        ft_model, tokenizer = self.get_ft_model(model_id, peft_model_id)

        # Attention map 추출
        attentions, inputs, generated_ids = self.get_attention_map(f"<s>[INST] {prompt} [/INST]", ft_model, tokenizer)

        # Input tokens
        input_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs['input_ids'][0]]

        # tokens 형태 출력 확인
        for token in input_tokens:
            self.logger.debug(token)

        # attentions의 차원 출력
        self.print_attention(attentions, log_on_console=False)

        # 디렉토리 생성 (시간별로 폴더 생성)
        output_dir = f"output/{model_id.split('/')[-1]}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        # 첫 번째 레이어, 첫 번째 헤드의 attention map을 이미지로 저장
        for layer, attention in enumerate(tqdm(attentions[0])):
            for head in range(attention.size(1)):
                # layer+1과 head+1을 두 자리로 맞춰서 파일 이름 생성
                layer_str: str = str(layer + 1).zfill(2)
                head_str: str = str(head + 1).zfill(2)
                
                self.save_attention_map(
                    attention_map=attentions, 
                    tokens=input_tokens, 
                    layer=layer, 
                    head=head, 
                    filename=f"{output_dir}/attention_map_layer{layer_str}_head{head_str}.png"
                )



class AttentionVisualizationConCat(AttentionVisualization):
    def concatenate_inputs_outputs(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        concatenated_dict = {}
        
        # Loop over the keys and concatenate the tensor values
        for key in inputs.keys():
            if key in outputs:
                concatenated_dict[key] = torch.cat([inputs[key], outputs[key]], dim=1)  # Concatenating along the sequence dimension (dim=1)
            else:
                concatenated_dict[key] = inputs[key]
        
        return concatenated_dict

    def generate_output(self, prompt: str, model: PeftModel, tokenizer: AutoTokenizer) -> str:
        inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Text를 generate하고, generating하면서 attention map과 함께 출력
        with torch.no_grad():  # 추론(inference) 모드 사용; 자동 미분(gradient 계산; 역전파 X)을 비활성화
            outputs = model.generate(**inputs, max_length=100, return_dict_in_generate=True, output_attentions=True, output_scores=True)
            
        outputs = tokenizer.decode(outputs.sequences[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        self.logger.debug(f"{prompt = }\n{outputs = }")
        return outputs

    def get_attention_map(self, prompt: str, model: PeftModel, tokenizer: AutoTokenizer) -> Tuple[List[List[torch.Tensor]], Dict[str, torch.Tensor], List[int]]:
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
        completion: str = self.generate_output(prompt, model, tokenizer)
        
        inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs: Dict[str, torch.Tensor] = tokenizer(completion, return_tensors="pt").to(model.device)
        concatenates: Dict[str, torch.Tensor] = self.concatenate_inputs_outputs(inputs, outputs)
        
        # Concatenating the actual tensors instead of BatchEncoding
        input_ids = inputs['input_ids']
        output_ids = outputs['input_ids']
        concatenated_ids = torch.cat([input_ids, output_ids], dim=1)
        
        # Text를 generate하고, generating하면서 attention map과 함께 출력
        with torch.no_grad():
            outputs = model.generate(input_ids=concatenated_ids, max_length=256, return_dict_in_generate=True, output_attentions=True, output_scores=True)
        
        # output_attentions=True로 설정했으므로, outputs.attentions에 attention weights가 포함됨
        attentions: List[List[torch.Tensor]] = outputs.attentions  # 각 레이어의 attention weights 리스트
        generated_ids: List[int] = outputs.sequences  # 생성된 텍스트에 해당하는 토큰 ID 목록
        return attentions, concatenates, generated_ids

    def print_attention(self, attentions: List[torch], log_on_console: bool = True) -> None:
        if log_on_console:
            for layer, attention in enumerate(attentions):
                for head in range(attention[0].size(1)):  # head 차원을 순회
                    attn_values = attention[head].cpu().numpy()

                    # 정규화: (x - min) / (max - min)
                    attn_min = np.min(attn_values)
                    attn_max = np.max(attn_values)
                    attn_norm = (attn_values - attn_min) / (attn_max - attn_min)

                    # 소수점 자릿수 조정 (예: 4자리까지)
                    attn_norm = np.round(attn_norm*100, 0)

                    tmp = f"Layer {layer+1}, Head {head+1} normalized attention:\n{attn_norm}\n"

                    # 값 출력
                    self.logger.info(tmp)

                    with open("tmp.txt", 'w', encoding="utf-8") as f:
                        f.writelines(tmp)
                    break
                break

    def __call__(self, prompt: Optional[str] = "일용직 작업자가 공사현장에서 비계3층 높이에서 외부비계 해체 작업중 발을 헛디뎌 추락함. 사고당일 병원으로 이송되었으며, 의식불명으로 입원중 사망함", model_id: Optional[str] = "beomi/Llama-3-Open-Ko-8B", peft_model_id: Optional[str] = "./models/Llama-3-Open-Ko-8B-csi-report-acctyp") -> None:
        ft_model, tokenizer = self.get_ft_model(model_id, peft_model_id)

        # Attention map 추출
        attentions, inputs, generated_ids = self.get_attention_map(f"<s>[INST] {prompt} [/INST]", ft_model, tokenizer)

        # Input tokens
        input_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs['input_ids'][0]]

        # Output tokens 수정: 입력 토큰 이후에 생성된 토큰을 올바르게 디코딩
        output_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in generated_ids[0][len(inputs['input_ids'][0]):]]

        # 입력 토큰과 출력 토큰 결합 (tokens를 올바르게 결합)
        tokens = input_tokens + output_tokens

        # tokens 형태 출력 확인
        for token in tokens:
            self.logger.debug(token)

        # attentions의 차원 출력
        self.print_attention(attentions, log_on_console=False)

        # 디렉토리 생성 (시간별로 폴더 생성)
        output_dir = f"output/{model_id.split('/')[-1]}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        # 첫 번째 레이어, 첫 번째 헤드의 attention map을 이미지로 저장
        for layer, attention in enumerate(tqdm(attentions[0])):
            for head in range(attention.size(1)):
                # layer+1과 head+1을 두 자리로 맞춰서 파일 이름 생성
                layer_str: str = str(layer + 1).zfill(2)
                head_str: str = str(head + 1).zfill(2)
                
                self.save_attention_map(
                    attention_map=attentions, 
                    tokens=input_tokens, 
                    layer=layer, 
                    head=head, 
                    filename=f"{output_dir}/attention_map_layer{layer_str}_head{head_str}.png"
                )

