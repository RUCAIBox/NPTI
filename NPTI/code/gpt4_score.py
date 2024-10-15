import os
import json
import tqdm
from openai import OpenAI

# 定义 OpenAI 客户端
api_key="sk-YUO9Cfl2ryAMJSwl7dBd8360D36f407b8a28Ba3859Bd61Fd"
url="https://aigc.x-see.cn/v1"
client = OpenAI(
    api_key=api_key,
    base_url=url
)
def get_prompt(mode,BFI,factors,question,answer):
    if mode=="R":
        prompt = """[System]
        Please act as an impartial judge and evaluate the fluency of the provided answer within a conversation. Focus solely on how smoothly and naturally the thoughts and emotions are conveyed,ignoring aspects like informal language use and the length of the response.  \
        Begin with a concise, objective explanation (no more than 30 words), and then assign a rating on a scale of 1 to 5, where a higher score indicates better fluency.
        Format your rating as follows: "Rating: [[rating]]" (e.g., "Rating: [[5]]").
        [The Start of the Answer]
        {answer}
        [The End of the Answer]
        """
        return prompt.format(answer=answer)
    else:
        prompt = """[System] 
        Please act as an impartial judge and evaluate the personality trait of {BFI} exhibited in the response provided by an AI assistant to the user question displayed below.\
        Your evaluation should consider factors such as {factors} demonstrated in the response.\
        Begin your evaluation by providing a short explanation of no more than 30 words. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 5,where a higher score indicates a stronger reflection of the specified trait.
        Please follow this format for your rating: "[[rating]]", for example: "Rating: [[5]]".
        [Question]
        {question}
        [The Start of Assistant’s Answer]
        {answer}
        [The End of Assistant’s Answer]
        """
        return prompt.format(BFI=BFI,factors=factors[BFI],question=question, answer=answer)
# 定义生成 prompt 的函数
def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

# 定义读取数据的函数
def get_data(filepath):
    datas = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                data = json.loads(line)
                datas.append(data)
    return datas


# 定义处理文件的主函数
def process_files(input_folder, output_folder, specified_files,BFI,factors):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 只处理指定的文件
    for file in specified_files:
        input_filepath = os.path.join(input_folder, file)
        output_filepath = os.path.join(output_folder, file)
        
        datas = get_data(input_filepath)
        
        with open(output_filepath, 'w', encoding='utf-8') as output_file:
            for i in tqdm.tqdm(range(len(datas))):
                question = datas[i]['question']
                answer = datas[i]['answer']
                questions=get_prompt(mode,BFI,factors,question,answer)
                ans = get_completion(questions, model="gpt-4o")
                t = {"question": question, "answer": answer, "score": ans}
                json.dump(t, output_file, ensure_ascii=False)
                output_file.write("\n")

# 设置文件夹路径
BFI="Openness"#Extraversion Neuroticism
factors={
    "Openness":"\"imagination\",\"artistic\",\"interests\",\"emotionality\",\"adventurousness\",\"intellect\" and liberalism\"",
    "Conscientiousness":"\"self-efficacy\",\"orderliness\",\"dutifulness\",\"achievement-striving\",\"self-discipline\",\"cautiousness\"",
    "Extraversion":"\"friendliness\",\"gregariousness\",\"assertiveness\",\"activity level\",\"excitement-seeking\" and \"cheerfulness\"",
    "Agreeableness":"\"trust\",\"morality\",\"altruism\",\"cooperation\",\"modesty\",\"sympathy\"",
    "Neuroticism":"\"anxiety\",\"anger\",\"depression\",\"self-consciousness\",\"immoderation\",\"vulnerability\""
}

for BFI in ["Conscientiousness","Openness","Extraversion","Agreeableness","Neuroticism"]:#
    for mode in ["","R"]:
        input_folder = f'NPTI/answer_results_cn/{BFI}'
        output_folder = f'NPTI/answer_results_cn/gpt4_score_{BFI}_{mode}'
        os.makedirs(output_folder, exist_ok=True)
        specified_files ='all'#[f'{BFI}_reversed_0.7.json',f'{BFI}_0.7.json']
        if specified_files == 'all':
            specified_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        process_files(input_folder, output_folder, specified_files,BFI,factors)