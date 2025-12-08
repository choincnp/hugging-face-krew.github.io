---
layout: post
title: "OpenAI의 새로운 오픈소스 모델 패밀리, GPT OSS를 환영합니다!"
author: youngjun
categories: [OpenSource, Model]
image: assets/images/blog/posts/2025-11-3-Welcome-GPT-OSS/thumbnail.png
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Welcome GPT OSS, the new open-source model family from OpenAI!](https://huggingface.co/blog/welcome-openai-gpt-oss)를 한국어로 번역한 글입니다._

---
# 들어가며
GPT OSS는 OpenAI가 공개한 대망의 오픈 가중치(open-weights) 모델로, 강력한 추론 능력과 에이전트 작업, 그리고 다양한 개발자의 사용 사례를 위해 설계되었습니다. 

이 모델은 두 가지 버전으로 구성되었습니다: 
- 117B 파라미터의 대형 모델([gpt-oss-120b](https://hf.co/openai/gpt-oss-120b))
- 21B 파라미터의 소형 모델([gpt-oss-20b](https://hf.co/openai/gpt-oss-20b))

두 모델 모두 혼합 전문가(Mixture-of-Experts, MoEs) 구조이며 4비트 양자화 방식(MXFP4)을 사용했기 때문에, 활성화되는 파라미터가 적어 빠른 추론이 가능하면서도 리소스 사용량은 낮게 유지됩니다. 대형 모델(gpt-oss-120b)은 H100 GPU 한 장에 올릴 수 있고, 소형 모델(gpt-oss-20b)은 16GB 메모리 내에서 구동되어 소비자용 하드웨어와 온디바이스 애플리케이션에 알맞습니다.

커뮤니티에 더 나은 영향력을 제공하기 위해, 이 모델들은 최소한의 사용 정책과 함께 Apache 2.0 라이선스로 공개되었습니다:

> 우리는 도구가 안전하고 책임감 있게, 그리고 민주적으로 사용되기를 바라며, 동시에 사용 방식에 대한 여러분의 통제권을 최대화하고자 합니다. gpt-oss를 사용함으로써 모든 관련 법규를 준수하는 데 동의하게 됩니다.

AI의 이점을 널리 접근할 수 있도록 하겠다는 사명에 따라 이번 릴리스는 오픈 소스 생태계에 대한 OpenAI의 헌신에 의미 있는 단계입니다. 많은 사용 사례가 프라이빗하거나 로컬 배포에 의존하는데, Hugging Face는 [OpenAI](https://huggingface.co/openai)가 커뮤니티에 합류하게 되어 무척 기쁩니다. 우리는 이 모델들이 오래 지속되고 영감을 주며 영향력 있는 모델이 될 것이라고 믿습니다.

# 기능 및 아키텍처 개요
- 총 파라미터는 21B와 117B, 활성 파라미터는 각각 3.6B와 5.1B
- MoE 가중치에만 적용된 [mxfp4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 포맷을 사용한 4비트 양자화로, 120B 모델은 80GB GPU 한 장에, 20B 모델은 16GB GPU 한 장에 적합
- Chain-of-Thought 및 조정 가능한 추론 강도 레벨을 지원하는 추론과 텍스트 전용 모델
- Instruction following 및 tool use 지원
- Transformers, vLLM, llama.cpp, ollama를 통한 로컬 실행 지원
- 추론에는 [Responses API](https://platform.openai.com/docs/api-reference/responses) 권장
- 라이선스: Apache 2.0, 소규모 보완 사용 정책 포함

**아키텍처**
- SwiGLU activations이 적용된 Token-choice MoE
- MoE 가중치 계산 시 선택된 전문가(experts)에 대해 softmax 적용 (softmax-after-topk)
- 각 어텐션 레이어는 128K 컨텍스트에 RoPE 사용
- 전체 컨텍스트와 128 토큰 슬라이딩 윈도우를 가지는 어텐션 레이어 교차 배치
- 어텐션 레이어는 헤드당 학습된 어텐션 싱크 사용 (softmax 분모에 추가적인 가산 값 적용)
- GPT-4o 및 기타 OpenAI API 모델과 동일한 토크나이저 사용
- Responses API와의 호환성을 위해 일부 새로운 토큰 추가

![](https://i.imgur.com/6Kc55rS.png)
`o3`, `o4-mini`와 비교한 OpenAI GPT OSS model들의 벤치마크 결과([OpenAI](https://openai.com/open-models/)제공).

# Inference Providers를 통한 API 액세스
OpenAI GPT OSS 모델은 Hugging Face의 [Inference Providers](https://huggingface.co/docs/inference-providers/en/index) 서비스를 통해 접근할 수 있어, 동일한 자바스크립트나 파이썬 코드로 지원되는 모든 프로바이더에 요청을 보낼 수 있습니다. 이것은 OpenAI 공식 데모인 [gpt-oss.com](http://gpt-oss.com)을 구동하는 것과 동일한 인프라이며, 여러분의 프로젝트에서도 활용할 수 있습니다.

다음은 Python과 매우 빠른 Cerebras provider를 사용하는 예제입니다. 더 많은 정보와 추가 예제는 [모델 카드의 inference providers 섹션](https://huggingface.co/openai/gpt-oss-120b?inference_api=true&inference_provider=auto&language=python&client=openai)과 [gpt-oss 모델들을 위한 가이드](https://huggingface.co/docs/inference-providers/guides/gpt-oss)를 확인하세요.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:cerebras",
    messages=[
        {
            "role": "user",
            "content": "How many rs are in the word 'strawberry'?",
        }
    ],
)

print(completion.choices[0].message)
```

추론 프로바이더는 OpenAI 호환 Responses API도 구현하고 있는데, 이는 채팅 모델을 위한 가장 진보된 OpenAI 인터페이스로, 더 유연하고 직관적인 상호작용을 위해 설계되었습니다.
다음은 Fireworks AI 프로바이더와 Responses API를 사용하는 예제입니다. 자세한 내용은 오픈소스 [responses.js](https://github.com/huggingface/responses.js) 프로젝트를 확인하세요.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-20b:fireworks-ai",
    input="How many rs are in the word 'strawberry'?",
)
print(response)
```

# 로컬 실행

## Transformers 사용하기
최신 transformers 릴리스(v4.55.1 이상)와 함께 `accelerate`, `kernels`를 설치해야 합니다. 또한 triton 3.4 이상을 설치하는 것을 권장하는데, 이를 통해 CUDA 하드웨어에서 mxfp4 양자화를 지원할 수 있습니다.

```bash
pip install --upgrade accelerate transformers kernels "triton>=3.4"
```

모델 가중치는 `mxfp4` 포맷으로 양자화되어 있으며, Hopper나 Blackwell 패밀리의 GPU와 호환됩니다. 이 형식은 원래 Hopper나 Blackwell 계열의 GPU에서만 사용 가능했지만, 이제는 이전 CUDA 아키텍처(Ada, Ampere, Tesla 포함)에서도 작동합니다. 여기에는 H100, H200, GB200 같은 데이터센터 카드와 50xx 시리즈 최신 소비자용 GPU가 포함됩니다. triton 3.4를 `kernels` 라이브러리와 함께 설치하면, 첫 사용 시 최적화된 `mxfp4` 커널을 다운로드하여 메모리를 크게 절약할 수 있습니다. 이러한 구성이 갖춰지면 16GB RAM을 가진 GPU에서도 20B 모델을 실행할 수 있습니다. 여기에는 많은 소비자용 그래픽 카드(3090, 4090, 5080)는 물론 Colab과 Kaggle도 포함됩니다!

테스트 결과 Triton 3.4는 최신 PyTorch 버전(2.7.x)에서 잘 작동합니다. 선택적으로 PyTorch 2.8을 설치할 수도 있는데, 작성 시점 기준으로는 프리릴리스 버전이지만([곧 정식 출시 예정](https://github.com/pytorch/pytorch/milestone/53)) triton 3.4와 함께 준비되어 안정적으로 작동합니다. PyTorch 2.8(triton 3.4 포함)과 triton 커널 설치 방법은 다음과 같습니다:

이전에 언급한 라이브러리들이 설치되어 있지 않거나 호환되는 GPU가 없는 경우, 모델은 양자화된 가중치에서 언팩된 `bfloat16` 형식으로 로드됩니다.

다음은 20B 모델로 간단한 추론을 실행하는 코드 예제입니다. 앞서 설명한 것처럼, `mxfp4`를 사용하면 16GB GPU에서 실행되며, `bfloat16`에서는 약 48GB가 필요합니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

### Flash Attention 3
이 모델들은 attention sinks 기법을 사용하는데, 이는 vLLM 팀이 Flash Attention 3와 호환되도록 만든 기법입니다. 우리는 그들의 최적화된 커널을 [kernels-community/vllm-flash-attn3](https://huggingface.co/kernels-community/vllm-flash-attn3)에 패키징하고 통합했습니다. 작성 시점 기준으로 이 초고속 커널은 PyTorch 2.7과 2.8에서 Hopper 카드로 테스트되었습니다. Hopper 카드(예: H100 또는 H200)에서 모델을 실행하는 경우, `pip install --upgrade kernels`를 실행하고 코드에 다음 줄을 추가해야 합니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+   # Flash Attention with Sinks
+   attn_implementation="kernels-community/vllm-flash-attn3",
)

messages = [
    {
    "role": "user", 
    "content": "How many rs are in the word 'strawberry'?"
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

이 코드는 [이전 블로그 포스트](https://huggingface.co/blog/hello-hf-kernels)에서 설명한 대로 `kernels-community`에서 최적화되고 프리컴파일된 커널 코드를 다운로드합니다. transformers 팀이 코드를 빌드하고 패키징하고 테스트했으니 안심하고 사용하셔도 됩니다.

### 기타 최적화
Hopper GPU 이상이 있다면 위에서 설명한 이유로 `mxfp4` 사용을 권장합니다. Flash Attention 3도 사용할 수 있다면 반드시 활성화하세요!

> 당신의 GPU가 `mxfp4`와 호환되지 않는다면 MegaBlocks MoE 커널을 사용해서 좋은 속도 향상을 얻을 수 있습니다. 다음과 같이 코드를 수정하면 됩니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+   # 다운로드 가능한 MegaBlocksMoeMLP로 MoE 레이어 최적화
+   use_kernels=True,
)

messages = [
    {
    "role": "user", 
    "content": "How many rs are in the word 'strawberry'?"
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

> MegaBlocks 최적화 MoE 커널은 모델이 `bfloat16`으로 실행되어야 하므로 `mxfp4`보다 메모리 소비가 높습니다. 가능하면 `mxfp4`를 사용하고, 그렇지 않다면 `use_kernels=True`로 MegaBlocks를 선택하세요.

### AMD ROCm 지원
OpenAI GPT OSS는 AMD Instinct 하드웨어에서 검증되었으며, `kernels` 라이브러리에서 AMD의 ROCm 플랫폼에 대한 초기 지원을 발표하게 되어 기쁩니다. 이는 향후 Transformers에서 최적화된 ROCm 커널을 제공하기 위한 기반을 마련합니다. MegaBlocks MoE 커널 가속은 AMD Instinct(예: MI300 시리즈)의 OpenAI GPT OSS에서 이미 사용할 수 있어, 더 나은 학습 및 추론 성능을 제공합니다. 위에 제시된 동일한 추론 코드로 테스트해볼 수 있습니다.

AMD는 사용자들이 AMD 하드웨어에서 모델을 시험해볼 수 있도록 Hugging Face [Space](https://huggingface.co/spaces/amd/gpt-oss-120b-chatbot)도 준비했습니다.

### 사용 가능한 최적화 요약
작성 시점 기준으로, 이 표는 GPU 호환성과 테스트를 기반으로 한 권장사항을 요약합니다. Flash Attention 3(sink attention 포함)이 추가 GPU와 호환되게 될 것으로 예상됩니다.

|                         | mxfp4                     | Flash Attention 3 (sink attention이 적용된) | MegaBlocks MoE kernels |
| ----------------------- | ------------------------- | --------------------------------------- | ---------------------- |
| Hopper GPU (H100, H200) | ✅                         | ✅                                       | ❌                      |
| 16GB RAM 이상의 CUDA GPU   | ✅                         | ❌                                       | ❌                      |
| 기타 CUDA GPU             | ❌                         | ❌                                       | ✅                      |
| AMD Instinct (MI3XX)    | ❌                         | ❌                                       | ✅                      |
| 활성화 방법                  | triton 3.4 + triton 커널 설치 | kernels-community의 vllm-flash-attn3 사용  | `use_kernels`          |

120B 모델이 H100 GPU 한 장에 들어가긴 하지만(`mxfp4` 사용 시), `accelerate`나 `torchrun`을 사용해서 여러 GPU에서도 쉽게 실행할 수 있습니다. Transformers는 기본 병렬화 플랜을 제공하며, 최적화된 어텐션 커널도 활용할 수 있습니다. 다음 스니펫은 GPU 4개가 있는 시스템에서 `torchrun --nproc_per_node=4 generate.py`코드와 함께 실행할 수 있습니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
import torch

model_path = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

device_map = {
    "tp_plan": "auto",  # Tensor Parallelism 활성화
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="kernels-community/vllm-flash-attn3",
    **device_map,
)

messages = [
    {"role": "user", "content": "Explain how expert parallelism works in large language models."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)

# 디코딩 및 출력
response = tokenizer.decode(outputs[0])
print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())
```

OpenAI GPT OSS 모델은 추론 과정의 일부로 도구 사용을 활용하도록 광범위하게 학습되었습니다. transformers를 위해 제작한 채팅 템플릿은 많은 유연성을 제공하니, 이 포스트의 뒤에 있는 `Transformers를 사용한 도구 사용` 섹션을 확인해보세요.

## Llama.cpp
Llama.cpp는 Flash Attention과 함께 MXFP4를 네이티브로 지원을 제공하여, Metal, CUDA, Vulkan 등 다양한 백엔드에서 최적의 성능을 첫 출시일부터 바로 제공합니다.

설치하려면 llama.cpp Github 저장소의 [가이드](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)를 따르세요.

```shell
# MacOS
brew install llama.cpp

# Windows
winget install llama.cpp
```

llama-server를 통해 사용하는 것이 권장됩니다:

```shell
llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 -fa --jinja --reasoning-format none
# 그런 다음 http://localhost:8080 에 접속
```

120B와 20B 모델 모두 지원합니다. 자세한 정보는 [이 PR](https://github.com/ggml-org/llama.cpp/pull/15091)이나 [GGUF 모델 컬렉션](https://huggingface.co/collections/ggml-org/gpt-oss-68923b60bee37414546c70bf)을 참조하세요.

## vLLM
언급한 대로, vLLM은 sink attention을 지원하는 최적화된 Flash Attention 3 커널을 개발했으므로 Hopper 카드에서 최고의 결과를 얻을 수 있습니다. Chat Completion과 Responses API 모두 지원됩니다. 다음 스니펫으로 설치하고 서버를 시작할 수 있는데, H100 GPU 2개를 사용한다고 가정합니다:

```bash
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2
```

또는 다음처럼 Python에서 직접 사용할 수 있습니다:

```python
from vllm import LLM

llm = LLM("openai/gpt-oss-120b", tensor_parallel_size=2)
output = llm.generate("San Francisco is a")
```

## `transformers serve`
[transformers serve](https://huggingface.co/docs/transformers/main/serving)를 사용해서 다른 의존성 없이 모델을 로컬에서 실험할 수 있습니다. 다음 명령으로 서버를 시작할 수 있습니다:

```bash
transformers serve
```

그런 다음, [Responses API](https://platform.openai.com/docs/api-reference/responses)를 사용해서 요청을 보낼 수 있습니다.

```bash
# responses API
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "system", "content": "hello"}], "temperature": 1.0, "stream": true, "model": "openai/gpt-oss-120b"}'
```

표준 Completions API로도 요청을 보낼 수 있습니다:

```bash
# completions API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 1.0, "max_tokens": 1000, "stream": true, "model": "openai/gpt-oss-120b"}'
```

# 파인튜닝
GPT OSS 모델은 `trl`과 완전히 통합되어 있습니다. Hugging Face에서는 `SFTTrainer`를 사용한 몇 가지 파인튜닝 예제를 개발했습니다:
- [OpenAI cookbook의 LoRA 예제](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) - 모델을 여러 언어로 추론하도록 파인튜닝하는 방법을 보여줍니다.
- [기본 파인튜닝 스크립트](https://github.com/huggingface/gpt-oss-recipes/blob/main/sft.py) - 필요에 맞게 수정할 수 있습니다.

# Hugging Face 파트너에 배포하기

## Azure
Hugging Face는 Azure와 협력하여 Azure AI Model Catalog를 이용해, 텍스트, 비전, 음성, 멀티모달 작업을 아우르는 
인기 있는 오픈소스 모델들을 고객 환경에 직접 제공합니다. 이를 통해 Azure의 엔터프라이즈급 인프라, 자동 확장, 모니터링을 활용하여 관리형 온라인 엔드포인트에 안전하게 배포할 수 있습니다.

GPT OSS 모델은 이제 Azure AI Model Catalog([GPT OSS 20B](https://ai.azure.com/explore/models/openai-gpt-oss-20b/version/1/registry/HuggingFace), [GPT OSS 120B](https://ai.azure.com/explore/models/openai-gpt-oss-120b/version/1/registry/HuggingFace))에서 사용 가능하며, 실시간 추론을 위한 온라인 엔드포인트에 바로 배포할 수 있습니다.

![](https://i.imgur.com/ywHYxcN.png)

## Dell
Dell Enterprise Hub는 Dell 플랫폼을 사용하여 최신 오픈 AI 모델을 온프레미스에서 훈련하고 배포하는 과정을 간소화하는 보안 온라인 포털입니다. Dell과 협력하여 개발된 이 허브는 최적화된 컨테이너, Dell 하드웨어에 대한 네이티브 지원, 엔터프라이즈급 보안 기능을 제공합니다.

GPT OSS 모델은 이제 [Dell Enterprise Hub](https://dell.huggingface.co/)에서 사용 가능하며, Dell 플랫폼을 사용해서 온프레미스에 바로 배포할 수 있습니다.

![](https://i.imgur.com/Pcva15s.png)


# 모델 평가하기
GPT OSS 모델은 추론 모델입니다. 따라서 평가 시 매우 큰 생성 크기(최대 새 토큰 수)가 필요합니다. 모델의 생성 결과에는 먼저 추론 과정이 포함되고, 그 다음에 실제 답변이 나오기 때문입니다. 생성 크기가 너무 작으면 추론 중간에 예측이 중단될 위험이 있으며, 이는 위음성을 발생시킬 수 있습니다. 메트릭을 계산하기 전에 모델 답변에서 추론 과정을 제거해야 하는데, 특히 수학이나 instruction 평가에서 파싱 오류를 방지하기 위해 필수적입니다.

다음은 lighteval로 모델을 평가하는 예제입니다(소스에서 설치해야 합니다).

```bash
git clone https://github.com/huggingface/lighteval
pip install -e .[dev] # 올바른 transformers 버전이 설치되었는지 확인하세요!

lighteval accelerate \
  "model_name=openai/gpt-oss-20b,max_length=16384,skip_special_tokens=False,generation_parameters={temperature:1,top_p:1,top_k:40,min_p:0,max_new_tokens:16384}" \
  "extended|ifeval|0|0,lighteval|aime25|0|0" \
  --save-details --output-dir "openai_scores" \
  --remove-reasoning-tags --reasoning-tags="[('<|channel|>analysis<|message|>','<|end|><|start|>assistant<|channel|>final<|message|>')]"
```

20B 모델의 경우 IFEval(strict prompt)에서 69.5 (+/-1.9), AIME25에서 63.3 (+/-8.9, pass@1)의 점수를 받아야 하는데, 이는 이 크기의 추론 모델로서 예상 범위 내의 점수입니다.

커스텀 평가 스크립트를 작성하려면, 추론 태그를 제대로 필터링하기 위해 토크나이저에서 `skip_special_tokens=False`를 사용해야 합니다. 이렇게 해야 모델 출력에서 전체 추론 과정을 얻을 수 있으며(위 예제와 동일한 문자열 쌍을 사용하여 추론을 필터링), 그 이유는 아래에서 확인할 수 있습니다.

# 채팅 및 채팅 템플릿
OpenAI GPT OSS는 출력에서 "채널(channel)" 개념을 사용합니다. 대부분의 경우 최종 사용자에게 보내지 않는 것(예: chain of thought)을 포함하는 "analysis" 채널과, 실제로 사용자에게 표시되도록 의도된 메시지를 포함하는 "final" 채널을 볼 수 있습니다.

도구가 사용되지 않는다고 가정할 때, 모델 출력 구조는 다음과 같습니다:
```
<|start|>assistant<|channel|>analysis<|message|>CHAIN_OF_THOUGHT<|end|><|start|>assistant<|channel|>final<|message|>ACTUAL_MESSAGE
```

대부분의 경우, `<|channel|>final<|message|>` 이후의 텍스트를 제외한 모든 것을 무시해야 합니다. 이 텍스트만 어시스턴트 메시지로 채팅에 추가하거나 사용자에게 표시해야 합니다. 다만 이 규칙에는 두 가지 예외가 있습니다: **훈련 중**이거나 모델이 **외부 도구를 호출**하는 경우에는 히스토리에 analysis 메시지를 포함해야 할 수 있습니다.

**훈련 시:**
훈련용 예제를 포맷할 때는 일반적으로 최종 메시지에 사고 과정(chain of thought)을 포함하고 싶을 것입니다. 이를 위한 올바른 위치는 `thinking` 키입니다.

```python
chat = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Can you think about this one?"},
    {"role": "assistant", "thinking": "Thinking real hard...", "content": "Okay!"}
]

# add_generation_prompt=False는 일반적으로 학습에만 사용되고, 추론에는 사용되지 않음
inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=False)
```

이전 턴에 thinking 키를 포함하거나 훈련이 아닌 추론을 할 때도 포함해도 되지만, 일반적으로 무시됩니다. 채팅 템플릿은 가장 최근의 사고 과정만 포함하며, 훈련 시에만(즉, `add_generation_prompt=False`이고 마지막 턴이 어시스턴트 턴일 때) 포함합니다.

이렇게 하는 이유는 미묘합니다: OpenAI gpt-oss 모델은 마지막 사고 과정을 제외한 모든 것이 제거된 멀티턴 데이터로 학습되었습니다. 즉, OpenAI `gpt-oss` 모델을 파인튜닝하려면 동일하게 해야 합니다.

- 채팅 템플릿이 마지막 것을 제외한 모든 사고 과정을 제거하도록 허용
- 마지막 어시스턴트 턴을 제외한 모든 턴의 레이블을 마스킹해야 합니다. 그렇지 않으면 사고 과정 없이 이전 턴을 학습하게 되어 모델이 사고 과정 없이 응답을 생성하도록 가르치게 됩니다. 이는 전체 멀티턴 대화를 단일 샘플로 훈련할 수 없다는 것을 의미합니다. 대신 어시스턴트 턴당 하나의 샘플로 나누어야 하며, 매번 마지막 어시스턴트 턴만 언마스킹 상태로 두어야 합니다. 이렇게 해야 모델이 각 턴에서 학습하면서도, 매번 마지막 메시지에서만 올바르게 사고 과정을 볼 수 있습니다.

## System 및 Developer 메시지
OpenAI GPT OSS는 채팅 시작 부분에서 "system" 메시지와 "developer" 메시지를 구분한다는 점에서 독특합니다. 하지만 대부분의 다른 모델은 "system"만 사용합니다. GPT OSS에서 system 메시지는 엄격한 형식을 따르며 현재 날짜, 모델 정체성, 사용할 추론 강도 같은 정보를 포함합니다. 반면 "developer" 메시지는 더 자유로운 형식이어서 (매우 혼란스럽게도) 대부분의 다른 모델의 "system" 메시지와 유사합니다.

표준 API로 GPT OSS를 더 쉽게 사용할 수 있도록, 채팅 템플릿은 "system" 또는 "developer" 역할의 메시지를 developer 메시지로 취급합니다. 실제 system 메시지를 수정하려면 채팅 템플릿에 `model_identity` 또는 `reasoning_effort` 인자를 전달할 수 있습니다:

```python
chat = [
    {
	    "role": "system", 
	    "content": "This will actually become a developer message!"
	}
]

tokenizer.apply_chat_template(
    chat,
    model_identity="You are OpenAI GPT OSS.",
    reasoning_effort="high"  # 기본값은 "medium"이지만, "high"와 "low"도 허용
)
```

## Transformers를 사용한 도구 사용
GPT OSS는 두 가지 종류의 도구를 지원합니다: "내장" 도구인 browser와 python, 그리고 사용자가 제공하는 커스텀 도구입니다. 내장 도구를 활성화하려면 아래처럼 채팅 템플릿의 `builtin_tools` 인자에 이름 목록을 리스트로 전달하세요. 커스텀 도구를 전달하려면, `tools` 인자를 사용하여 JSON 스키마 형식으로 전달하거나, 타입 힌트와 docstring을 포함한 Python 함수로 전달할 수 있습니다. 자세한 내용은 [채팅 템플릿 도구 문서](https://huggingface.co/docs/transformers/en/chat_extras)를 참조하거나, 아래 예제를 수정하면 됩니다:

```python
def get_current_weather(location: str):
    """
    Returns the current weather status at a given location as a string.

    Args:
        location: The location to get the weather for.
    """
    return "Terrestrial."  # 좋은 날씨 도구라고는 안 했습니다

chat = [
    {"role": "user", "content": "What's the weather in Paris right now?"}
]

inputs = tokenizer.apply_chat_template(
    chat,
    tools=[weather_tool],
    builtin_tools=["browser", "python"],
    add_generation_prompt=True,
    return_tensors="pt"
)
```

모델이 도구를 호출하기로 선택하면(`<|call|>`로 끝나는 메시지로 표시), 채팅에 도구 호출을 추가하고, 도구를 호출한 다음, 도구 결과를 채팅에 추가하고 다시 생성해야 합니다:

```python
tool_call_message = {
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "arguments": {"location": "Paris, France"}
            }
        }
    ]
}
chat.append(tool_call_message)

tool_output = get_current_weather("Paris, France")

tool_result_message = {
    # GPT OSS는 한 번에 하나의 도구만 호출하므로,
    # tool 메시지에 추가 메타데이터가 필요하지 않습니다!
    # 템플릿이 이 결과가 가장 최근 도구 호출의 것임을 알아낼 수 있습니다.
    "role": "tool",
    "content": tool_output
}
chat.append(tool_result_message)

# 이제 apply_chat_template()과 generate()를 다시 실행할 수 있고,
# 모델은 대화에서 도구 결과를 사용할 수 있습니다.
```
