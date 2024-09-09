from flask import Flask, request, send_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import os

# Flask app setup
app = Flask(__name__)

# Model and tokenizer setup
model_id = "beomi/Llama-3-Open-Ko-8B"
peft_model_id = "results/checkpoint-5000"

config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()

def get_attention_map(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, return_dict_in_generate=True, output_attentions=True, output_scores=True)
    
    attentions = outputs.attentions
    generated_ids = outputs.sequences
    return attentions, inputs, generated_ids

def save_attention_map(attention_map, tokens, layer=0, head=0, filename="attention_map.png"):
    attention = attention_map[0][layer][0, head].cpu().numpy()
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)  
    ax.set_yticklabels(tokens, rotation=0, fontsize=8)
    plt.title(f"Layer {layer + 1}, Head {head + 1}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

@app.route('/generate', methods=['POST'])
def generate_attention_map():
    data = request.json
    prompt = data.get('prompt', '')

    attentions, inputs, generated_ids = get_attention_map(f"<s>[INST] {prompt} [/INST]")
    tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs['input_ids'][0]]

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    for layer, attention in enumerate(tqdm(attentions[0])):
        for head in range(attention.size(1)):
            layer_str = str(layer + 1).zfill(2)
            head_str = str(head + 1).zfill(2)
            filename = os.path.join(output_dir, f"attention_map_layer{layer_str}_head{head_str}.png")
            save_attention_map(attentions, tokens, layer=layer, head=head, filename=filename)
            image_paths.append(filename)
    
    return send_file(image_paths[0], mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
