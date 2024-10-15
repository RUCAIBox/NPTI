import argparse
from types import MethodType
from IPython import embed
import torch
import json
from tqdm import tqdm
import random
import gc
import os
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# Modify argument parsing to support shell script inputs
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model path")
parser.add_argument("--question_dir", type=str, required=True, help="Directory for questions")
parser.add_argument("--answer_dir", type=str, required=True, help="Directory for saving answers")
parser.add_argument("--neuron_dir", type=str, required=True, help="Directory for saving neuron results")
parser.add_argument("--personality_desc", type=str, required=True, help="Path to personality descriptions")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")

# Parse arguments
args = parser.parse_args()

# Define directories based on passed arguments
question_directory = args.question_dir
answer_directory = args.answer_dir
neuron_directory = args.neuron_dir
personality_description_path = args.personality_desc
batch_size = args.batch_size

# Ensure directories exist or create them
os.makedirs(answer_directory, exist_ok=True)
os.makedirs(neuron_directory, exist_ok=True)
TEMPLATE="""
You will find a personality description followed by a question below. I want you to forget who you are and fully immerse yourself in the persona described, adopting not only their perspective but also their tone and attitude. With this new identity in mind, please respond to the question.
Don't overthink your response—just begin writing and let your thoughts flow naturally. Spelling and grammar are not important here; what's essential is capturing the essence of this personality in your answer. Try to keep your response under 300 words.
###Personality description:{personality}

###Question:{question}

###Response:
"""

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
max_length = model.llm_engine.model_config.max_model_len  # 最大上下文长度
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers  # 层数->32
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size  # 中间层大小-
personality_data = {}
with open(personality_description_path, 'r') as file:
    for line in file:
        json_object = json.loads(line.strip())
        for key in json_object:
            if key in personality_data:
                personality_data[key].append(json_object[key])
            else:
                personality_data[key] = [json_object[key]]
for key, value in personality_data.items():
    assert len(value) == 80, f"Length of '{key}' list is not 80"

def load_question(question_path):
    question_data = []
    with open(question_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if data:
                question_data.append(data['question'])
    return question_data

def factory(idx): 
    def llama_forward(self, x): 
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()  # b,neuron
        if j<=batch_size:
            global token_num
            token_num+=j
            over_zero[idx, :] += (activation > 0).sum(dim=(0))
            bin_indices = (torch.bucketize(activation, bins) - 1).transpose(0, 1)
            increment_tensor = torch.ones_like(bin_indices, dtype=histograms[idx, 0].dtype)
            histograms[idx].scatter_add_(1, bin_indices, increment_tensor)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x

    return llama_forward

for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)  # 绑定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for BFI in ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]:
    question_data=load_question(question_path=f"{question_directory}/{BFI}.json")
    for mode in ["","_reversed"]:
        data_type = BFI + mode
        ans_list=[]
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda') 
        token_num=0 
        histograms = torch.zeros((num_layers, intermediate_size, 301)).cuda()
        bins = torch.cat([torch.tensor([-float('inf')]).to('cuda'), torch.arange(0, 3.01, 0.01).to('cuda')])
        bins[-1] = float('inf')
        num_bins = 301
        output_file_qa=f'{answer_directory}/{data_type}.json'
        output_file_neuron=f'{neuron_directory}/{data_type}.pt'
        with open(output_file_qa, 'w', encoding='utf-8') as output_file_qa:
            for i in tqdm(range(0, len(question_data), batch_size)):
                batch_questions = question_data[i:i+batch_size]
                input_texts = []
                for question in batch_questions:
                    personality = random.choice(personality_data[data_type])
                    input_text = TEMPLATE.format(personality=personality, question=question)
                    input_texts.append([{"role": "user", "content": input_text}])
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True)
                sampling_params = SamplingParams(
                    max_tokens=500,
                    temperature=0,
                    repetition_penalty=1.1
                )
                output = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
                ans_list =ans_list+[json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' for i in range(len(batch_questions))]
            output_file_qa.writelines(ans_list)
            output = dict(token_num=token_num/num_layers,question_num=len(question_data), over_zero=over_zero.to('cpu'),histograms=histograms.to('cpu'))
            torch.save(output, output_file_neuron) #output[1].outputs[0].text
        del over_zero, histograms, bins, ans_list, output
        torch.cuda.empty_cache()  # 清理GPU缓存
        gc.collect()