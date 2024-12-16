import  os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from zhipuai import ZhipuAI

from typing import Any, List
class ZhipuApi():
    """
    
    """

    client: Any
    """`zhipuai.ZhipuAI`"""
    def gen_glm_params(self, prompt):
        '''
        构造glm模型参数

        请求参数：
            prompt： 对应的用户提示词
        '''

        message = [
            {"role": "system", "content": "你是一个学识渊博的教授，需要对输入的学术问题作答，不可以回答国家政治相关问题"},
            {"role": "user", "content": prompt}
        ]
        return message

def get_completion(self, prompt, model='glm-4-plus', temperature=0.95):
    '''
    获取GLM模型调用结果

    请求参数：
        prompt:对应提示词
        model:调用的模型，默认为glm-4-plus
        temperature采样温度,控制输出的随机度
    '''
    message = self.gen_glm_params(prompt)
    response = self.client.chat.completions.create(
        model = model,
        messages = message,
        temperature = temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"