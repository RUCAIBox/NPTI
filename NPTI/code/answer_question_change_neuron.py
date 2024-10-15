import argparse
from types import MethodType
from IPython import embed
import torch
import json
import numpy as np
from tqdm import tqdm
import random
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
import ast
TEMPLATE="""
Imagine you are a real person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions. 

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:

"""
#!!!!!!!!!!!!!!!!
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, required=True, help="Path to the model")  # 模型路径
parser.add_argument( "--neuron_dir", type=str, required=True, help="Path to neuron results directory")
parser.add_argument( "--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument( "--question_dir", type=str, required=True, help="Path to the question directory")
parser.add_argument( "--batch_size", type=int, default=32, help="Batch size")
args = parser.parse_args()
neuron_dir = args.neuron_dir
output_dir = args.output_dir
question_dir = args.question_dir
batch_size = args.batch_size
#!!!!!!!!!!!!!!!!!!
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
# 读取问题描述
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_question(question_path):
    question_data = []
    with open(question_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if data:
                question_data.append(data['question'])
    return question_data
#读取要改的神经元
def load_neuron_to_change(neuron_to_change_path):
    with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
        neuron_to_change = json.loads(file.read())
        for t in neuron_to_change:
            neuron_to_change[t]=torch.tensor(neuron_to_change[t]).to(device)
    return neuron_to_change
def factory(idx): 
    def llama_forward(self, x): 
        def custom_function(x):
            return 1 / (1 + torch.exp(-10 * (x - 0.15)))
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])  # 过一个激活函数.neurn=self.act_fn(self.gate_proj(x))
        if j <= batch_size and str(idx) in neuron_to_change:
            elements = neuron_to_change[str(idx)]
            indices = elements[:, 1].long()
            values = elements[:, 4]
            difference=elements[:,2]
            thresholds=0.9
            random_tensor = torch.rand(len(values)).to(device)
            mask = random_tensor <=thresholds
            if str(idx) in neuron_to_deactivate:
                elements_to_deactivate=neuron_to_deactivate[str(idx)]
                indices_to_deactivate=elements_to_deactivate[:, 1].long()
                gate_up[:,indices_to_deactivate]=torch.min(gate_up[:, indices_to_deactivate], torch.tensor(0.0))
            gate_up[:,indices[mask]] += values[mask] *val*custom_function(difference[mask])#激活
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x

    return llama_forward

for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp ##
    obj.forward = MethodType(factory(i), obj)  # 绑定
for BFI in ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]:
    question_path =f'{question_dir}/{BFI}.json'
    for val in [0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
        for mode in["_reversed"]:#"_reversed",
            data_type = BFI + mode
            neuron_to_change_path=f'{neuron_dir}/{data_type}_dict.json'
            if mode=="_reversed":
                neuron_to_deactivate_path=f'{neuron_dir}/{BFI}_dict.json'
            if mode=="":
                neuron_to_deactivate_path=f'{neuron_dir}/{BFI}_reversed_dict.json'
            output_dir_bfi=f'{output_dir}/{BFI}'
            os.makedirs(output_dir_bfi, exist_ok=True)
            output_file=f'{output_dir_bfi}/{data_type}_{val}.json'
            with open(output_file, 'w', encoding='utf-8') as output_file:
                question_data=load_question(question_path)
                neuron_to_change=load_neuron_to_change(neuron_to_change_path)
                neuron_to_deactivate=load_neuron_to_change(neuron_to_deactivate_path)
                for i in tqdm(range(0, len(question_data), batch_size)):
                    num_of_words=0#初始化一下
                    batch_questions = question_data[i:i+batch_size]
                    input_texts = []
                    for question in batch_questions:
                        input_text =TEMPLATE.format(question=question)#TODO
                        input_texts.append([{"role": "user", "content": input_text}])
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True)
                    sampling_params = SamplingParams(
                        max_tokens=400,
                        temperature=0.0,
                        repetition_penalty=1.15
                    )
                    output = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
                    json_lines = [json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' 
                        for i in range(len(batch_questions))]
                    output_file.writelines(json_lines)