
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from zhipuai import ZhipuAI

# 使用继承
class ZhipuAILLM(LLM):
    # 默认使用glm-4-plus
    model: str = 'glm-4-plus'
    
    # 温度系数
    temperature: float = 0.1
    
    # api_key
    api_key: str = None
    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        client = ZhipuAI(
            api_key = self.api_key
        )
        
        def gen_plm_params(prompt):
            '''
            构造 GLM 模型请求参数 messages
            
            请求参数：
                prompt： 对应的用户提示词
            '''
            messages = [{"role": "user", "content": prompt}]
            return messages
        messages = gen_plm_params(prompt)
        response = client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature
        )
        
        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用API的默认参数"""
        normal_params = {
            "temperature": self.temperature
        }
        return {**normal_params}
    
    @property
    def _llm_type(self) -> str:
        return 'zhipu'
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the indetifying parameters."""
        return {**{"model": self.model}, **self._default_params}