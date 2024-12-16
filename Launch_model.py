import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import time
import torch
import cml.metrics_v1 as metrics
import cml.models_v1 as models

import sentencepiece

hf_access_token = os.environ.get('HF_ACCESS_TOKEN')

# Quantization
# Here quantization is setup to use "Normal Float 4" data type for weights. 
# This way each weight in the model will take up 4 bits of memory. 
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Create a model object with above parameters
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_access_token
)

# Args helper
def opt_args_value(args, arg_name, default):
  """
  Helper function to interact with LLMs parameters for each call to the model.
  Returns value provided in args[arg_name] or the default value provided.
  """
  if arg_name in args.keys():
    return args[arg_name]
  else:
    return default

# Define tokenizer parameters


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=hf_access_token,
    use_fast=True  # Explicitly request the fast tokenizer
)

tokenizer.pad_token = tokenizer.eos_token

# Mamke the call to 
def generate(prompt, max_new_tokens=50, temperature=0, repetition_penalty=1.0, num_beams=1, top_p=1.0, top_k=0):
  """
  Make a request to the LLM, with given parameters (or using default values).
    1. max_new_tokens
        설명: 생성될 최대 토큰(단어) 수를 제한합니다. 이 값에 도달하면 응답 생성을 중단합니다. 일반적으로 영어에서는 한 단어에 1.3 토큰, 한글은 한 단어에 1토큰
        목적: 응답 길이 조절.
    2. temperature
        설명: 생성의 무작위성을 제어합니다.
         - 0에 가까울수록 더 정형화된 응답을 생성합니다 (확실도가 높은 토큰을 선택).
         - 1에 가까울수록 더 창의적이고 다양한 응답이 나옵니다.
        목적: 생성 결과의 창의성 제어.
    3. repetition_penalty
        설명: 응답에 반복된 단어가 사용되는 것을 방지합니다.
         - 값이 1이면 반복에 대한 패널티가 없습니다.
         - 값이 >1이면 반복될 확률이 감소합니다.
        목적: 같은 단어나 문장이 계속 반복되는 문제 해결.
    4. num_beams (Beam Search)
        설명: 몇 개의 다른 토큰 시퀀스를 동시에 생성하여 최상의 결과를 선택할지 제어합니다.
         - num_beams=1: Greedy Search (단일 시퀀스).
         - num_beams > 1: 여러 시퀀스를 고려해 최적 결과를 선택합니다.
        목적: 생성 결과 품질 개선 (더 나은 답변 선택).
    5. top_p (Nucleus Sampling)
        설명: 누적 확률이 top_p에 도달할 때까지 가장 가능성이 높은 토큰을 선택합니다.
          - top_p=0.9는 상위 90% 확률에 해당하는 토큰만 고려합니다.
        목적: 확률이 낮은 토큰 제거 → 생성의 품질과 다양성 균형 유지.
    6. top_k
        설명: 가장 가능성이 높은 top_k개의 토큰만 고려합니다.
         -  top_k=50은 상위 50개 확률 토큰 중에서만 다음 토큰을 선택합니다.
        목적: 불필요한 토큰 제거 → 모델이 불확실할 때 더 정확한 출력을 만듭니다.
  """
  batch = tokenizer(prompt, return_tensors='pt').to("cuda")
  
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    top_p=top_p,
                                    top_k=top_k)
  
  output=tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  
  # Log the response along with parameters
  print("Prompt: %s" % (prompt))
  print("max_new_tokens: %s; temperature: %s; repetition_penalty: %s; num_beams: %s; top_p: %s; top_k: %s" % (max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k))
  print("Full Response: %s" % (output))
  
  return output

@models.cml_model(metrics=True)
def api_wrapper(args):
  """
  Process an incoming API request and return a JSON output.
  """
  start = time.time()
  
  # Pick up args from model api
  prompt = args["prompt"]
  
  # Pick up or set defaults for inference options
  # TODO: More intelligent control of max_new_tokens
  temperature = float(opt_args_value(args, "temperature", 0))
  max_new_tokens = float(opt_args_value(args, "max_new_tokens", 50))
  top_p = float(opt_args_value(args, "top_p", 1.0))
  top_k = int(opt_args_value(args, "top_k", 0))
  repetition_penalty = float(opt_args_value(args, "repetition_penalty", 1.2))
  num_beams = int(opt_args_value(args, "num_beams", 1))
  
  
  # Generate response from the LLM
  response = generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)
  
  # Calculate elapsed time
  response_time = time.time() - start
  
  # Track model outputs over time
  metrics.track_metric("prompt", prompt)
  metrics.track_metric("response", response)
  metrics.track_metric("response_time_s", response_time)

  
  return json.loads(response)['prediction']