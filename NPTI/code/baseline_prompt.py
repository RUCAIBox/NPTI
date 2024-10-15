import argparse
from types import MethodType
from IPython import embed
import torch
import json
import os
from tqdm import tqdm
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
#!!!!!!!!!!!!!!!
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)  # Model path
parser.add_argument("--personality_description", type=str)  # Personality description path
parser.add_argument("--output_folder", type=str)  # Output folder path
parser.add_argument("--question_dir", type=str)  # Output folder path
parser.add_argument("--batch_size", type=int)  # Batch size
args = parser.parse_args()
personality_description_path=args.personality_description
output_folder=args.output_folder
batch_size = args.batch_size
question_dir=args.question_dir
#!!!!!!!!!!!
TEMPLATE_sp="""
Imagine you are {personality} person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions. 

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:
"""
TEMPLATE_p2="""
Imagine you are a real person rather than a language model. {personality}.Now you're asked by the following question. Write your response based on your authentic thoughts and emotions. 

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:
"""

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
personality_adj={
    "Extraversion":"an extraverted",
    "Openness":"an open",
    "Conscientiousness":"a conscientious",
    "Agreeableness":"an agreeable",
    "Neuroticism":"a neurotic",
    "Extraversion_reversed":"an introverted",
    "Openness_reversed":"a closed",
    "Conscientiousness_reversed":"an unconscientious",
    "Agreeableness_reversed":"a disagreeable",
    "Neuroticism_reversed":"a calm"
}
def get_personality_data():
    personality_data = {}
    with open(personality_description_path, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            for key in json_object:
                if key in personality_data:
                    personality_data[key].append(json_object[key])
                else:
                    personality_data[key] = [json_object[key]]
    return personality_data
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
        j = x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2]) 
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x

    return llama_forward
for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)  # 绑定
for BFI in ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]:
    question_path =f'{question_dir}/{BFI}.json'
    for md in ["sp","p2"]:
        output_folder_path = f'{output_folder}/{md}/{BFI}'
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for mode in["","_reversed"]:
            personality_data=get_personality_data()
            data_type = BFI + mode
            output_file=f'{output_folder_path}/{data_type}.json'
            with open(output_file, 'w', encoding='utf-8') as output_file:
                question_data=load_question(question_path)
                for i in tqdm(range(0, len(question_data), batch_size)):
                    num_of_words=0#初始化一下
                    batch_questions = question_data[i:i+batch_size]
                    input_texts = []
                    for question in batch_questions:
                        personality_des = random.choice(personality_data[data_type])
                        if md=="sp":
                            input_text = TEMPLATE_sp.format(personality=personality_adj[data_type], question=question)
                        if md=="p2":
                            input_text = TEMPLATE_sp.format(personality=personality_des, question=question)
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