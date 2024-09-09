from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import urllib.request
import numpy as np
import torch

# 나눔고딕 폰트 다운로드 및 설정
"""
font_url = "https://github.com/naver/nanumfont/releases/download/v1.0/NanumGothic.ttf"
font_path = "/tmp/NanumGothic.ttf"
urllib.request.urlretrieve(font_url, font_path)
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
"""
# plt.rc('font', family="NanumGothicCoding")  # 'NanumGothicCoding')
# mpl.rcParams['axes.unicode_minus'] = False  # 위의 코드를 적용하면 마이너스 폰트가 깨지는 경우를 방지할 수 있다.
# plt.rcParams['font.family'] = 'NanumGothic'

model_id = "beomi/Llama-3-Open-Ko-8B"
peft_model_id = "results/checkpoint-5000"

config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
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

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()

def get_attention_map(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # outputs = model(**inputs, output_attentions=True)
        outputs = model.generate(**inputs, max_length=100, return_dict_in_generate=True, output_attentions=True, output_scores=True)
    
    # output_attentions=True로 설정했으므로, outputs.attentions에 attention weights가 포함됨
    attentions = outputs.attentions  # 각 레이어의 attention weights 리스트
    generated_ids = outputs.sequences
    return attentions, inputs, generated_ids

def save_attention_map(attention_map, tokens, layer=0, head=0, filename="attention_map.png"):
    attention = attention_map[layer][0, head].cpu().numpy()  # 특정 레이어와 헤드의 attention weights 선택
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

prompt = "알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?"  # "If a part-time worker quits after working for three days, do I still have to pay them for the weekly holiday allowance?"  # "알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?"

# Attention map 추출
attentions, inputs, generated_ids = get_attention_map(f"<s>[INST] {prompt} [/INST]")
# Tokenize된 텍스트를 가져옴
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# Token 단위로 디코딩
# tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs['input_ids'][0]]
# Tokenize된 텍스트를 복원
# tokens = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False)
# 
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

for tkn in tokens:
    print(tkn)

# 전체 텐서 출력 설정
np.set_printoptions(threshold=np.inf)
print(f"{attentions[0][0].size() = }")
# attentions의 차원 출력
for layer, attention in enumerate(attentions):
    for head in range(attention.size(1)):  # head 차원을 순회
        attn_values = attention[0, head].cpu().numpy()
        
        # 정규화: (x - min) / (max - min)
        attn_min = np.min(attn_values)
        attn_max = np.max(attn_values)
        attn_norm = (attn_values - attn_min) / (attn_max - attn_min)
        
        # 소수점 자릿수 조정 (예: 4자리까지)
        attn_norm = np.round(attn_norm*100, 0)
        
        tmp = f"Layer {layer+1}, Head {head+1} normalized attention:\n{attn_norm}\n"
        
        # 값 출력
        print(tmp)
        
        with open("tmp.txt", 'w', encoding="utf-8") as f:
            f.writelines(tmp)
        break
    break

# 첫 번째 레이어, 첫 번째 헤드의 attention map을 이미지로 저장
save_attention_map(attentions, tokens, layer=0, head=0, filename="output/attention_map_layer1_head1.png")
