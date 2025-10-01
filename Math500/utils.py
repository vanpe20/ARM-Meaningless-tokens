from ..modeling_qwen2 import Qwen2ForCausalLM
from ..modeling_llama import LlamaForCausalLM
from ..modeling_gemma3 import Gemma3ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


def choose_model( model, arm = True):
    if arm is True:
        if 'Qwen2.5' in model:
            model = Qwen2ForCausalLM.from_pretrained(model)
        elif 'Llama' in model:
            model = LlamaForCausalLM.from_pretrained(model)
        elif 'gemma' in model:
            model = Gemma3ForCausalLM.from_pretrained(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    
    return model


def choose_parameters(model):
    if 'Qwen2.5-Math-1.5B' in model:
        return 0.13, 99.5
    elif "Qwen/Qwen2.5-Math-7B" in model:
        return 0.1, 95
    elif "Qwen2.5-7B-Instruct" in model:
        return 0.1, 99.5
    elif 'Qwen2.5-32B-Instruct' in model:
        return None, None
    elif 'gemma-3-4b-it' in model:
        return 0.25, 0.85
    elif 'gemma-3-27b-it' in model:
        return 0.25, 0.85
    elif 'Llama-3.1-8B-Instruct' in model:
        return 0.32, 90
    else:
        raise ValueError(f"No parameters of model {model}, you need to define parameters by yourself.")