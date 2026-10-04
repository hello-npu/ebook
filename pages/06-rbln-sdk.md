# 06. 리벨리온 NPU 사용 예제

리벨리온(Rebellions)의 RBLN SDK가 미리 설치된 Python 가상환경에서 시작합니다. 가상환경을 활성화한 뒤 SDK 관련 패키지와 시스템 라이브러리, NPU 장치를 확인하고 간단한 텐서 연산 및 Python 유틸리티 API를 실행합니다.

참고 문서:

- [RBLN SDK v0.11.2 사용자 가이드](https://docs.rbln.ai/v0.11.2/ko/index.html)

## 1. 가상환경 활성화

SDK가 설치된 가상환경을 활성화합니다.

```bash
source ~/rbln-env/bin/activate
```

이후 명령은 해당 가상환경이 활성화된 상태에서 실행합니다.

## 2. 서버의 SDK 및 장치 확인

### 2-1. 환경변수 확인

`rbln` 문자열이 포함된 환경변수를 확인합니다.

```bash
env | grep -Ei 'rbln'
```

출력:

```bash
VIRTUAL_ENV=/home/hellonpu/rbln-env
VIRTUAL_ENV_PROMPT=(rbln-env)
```

이 출력에서는 활성화된 가상환경의 경로와 프롬프트 설정을 확인할 수 있습니다. SDK 설치 여부는 아래 패키지와 라이브러리 확인을 통해 별도로 점검합니다.

### 2-2. 설치된 관련 패키지 확인

```bash
pip list | grep -Ei 'rbln|rebel|torch'
```

출력:

```bash
optimum-rbln                             0.10.1
rebel-compiler                           0.10.1
torch                                    2.9.0+cpu
torch-rbln                               0.1.7
torchaudio                               2.9.0
torchvision                              0.24.0
vllm_rbln                                0.10.1.post2
```

### 2-3. 시스템 라이브러리 확인

시스템의 공유 라이브러리 캐시에서 리벨리온 관련 라이브러리를 확인합니다.

```bash
ldconfig -p 2>/dev/null | grep -Ei 'rbln|rebel'
```

출력:

```bash
        librbln-thunk.so.3 (libc6,x86-64) => /lib/librbln-thunk.so.3
        librbln-thunk.so (libc6,x86-64) => /lib/librbln-thunk.so
        librbln-ml.so (libc6,x86-64) => /lib/librbln-ml.so
        librbln-ccl.so.3 (libc6,x86-64) => /lib/librbln-ccl.so.3
        librbln-ccl.so (libc6,x86-64) => /lib/librbln-ccl.so
```

### 2-4. NPU 장치 확인

NVIDIA GPU에서 `nvidia-smi`로 장치 상태를 확인하는 것과 유사하게, 리벨리온 NPU에서는 `rbln-smi`를 사용합니다.

```bash
rbln-smi
```

출력:

```bash
Fri Oct 2 14:19:45 2026
+-------------------------------------------------------------------------------------------------+
|                                Device Information KMD ver: 3.0.0                                |
+-----+-----------+---------+---------------+------+---------+------+---------------------+-------+
| NPU |    Name   | Device  |   PCI BUS ID  | Temp |  Power  | Perf |  Memory(used/total) |  Util |
+=====+===========+=========+===============+======+=========+======+=====================+=======+
| 0   | RBLN-CA22 | rbln0   |  0000:24:00.0 |  38C |  25.2W  | P14  |    0.0B / 15.7GiB   |   0.0 |
+-----+-----------+---------+---------------+------+---------+------+---------------------+-------+
+-------------------------------------------------------------------------------------------------+
|                                       Context Information                                       |
+-----+---------------------+--------------+-----------+----------+------+---------------+--------+
| NPU | Process             |     PID      |    CTX    | Priority | PTID |      Memalloc | Status |
+=====+=====================+==============+===========+==========+======+===============+========+
| N/A | N/A                 |     N/A      |    N/A    |   N/A    | N/A  |           N/A |  N/A   |
+-----+---------------------+--------------+-----------+----------+------+---------------+--------+
```

출력에서 확인한 장치 상태는 다음과 같습니다.

| 항목 | 값 |
|---|---|
| NPU 개수 | 1개 |
| 장치 이름 | RBLN-CA22 |
| 장치 식별자 | rbln0 |
| KMD 버전 | 3.0.0 |
| 온도 | 38°C |
| 소비 전력 | 25.2W |
| 메모리 사용량 / 전체 용량 | 0.0B / 15.7GiB |
| 표시된 실행 컨텍스트 | 없음 |

## 3. 간단한 텐서 연산 실행

[공식 PyTorch RBLN 튜토리얼](https://docs.rbln.ai/latest/ko/software/rbln_pytorch/tutorial_running_n_debugging.html)의 `torch.add` 예제를 실행합니다.

`device="rbln"`을 지정해 NPU에 FP16 텐서를 생성하고, 두 텐서의 원소별 덧셈을 수행합니다. 코드의 `a + b`는 `torch.add`에 해당하는 연산입니다.

### 코드: test_add.py

```python
import torch

device = "rbln"

a = torch.tensor(, dtype=torch.float16, device=device)
b = torch.tensor(, dtype=torch.float16, device=device)

c = a + b
print(c)
```

### 실행 결과

```bash
tensor([5., 7., 9.], device='rbln:0', dtype=torch.float16)
```

결과 텐서의 값은 `[5., 7., 9.]`이며, 장치는 `rbln:0`, 데이터 타입은 `torch.float16`으로 표시됩니다.

## 4. Python 유틸리티 API

`rebel` 모듈의 유틸리티 API로 NPU 개수, 특정 장치의 사용 가능 여부, 장치 이름을 확인할 수 있습니다.

NVIDIA GPU에서 사용하는 PyTorch CUDA API와 기능상 대응되는 API는 다음과 같습니다.

| 확인 항목 | NVIDIA GPU: PyTorch CUDA | 리벨리온 NPU: rebel |
|---|---|---|
| 사용 가능 여부 | `torch.cuda.is_available()` | `rebel.npu_is_available(i)` |
| 장치 이름 | `torch.cuda.get_device_name(i)` | `rebel.get_npu_name(i)` |
| 장치 개수 | `torch.cuda.device_count()` | `rebel.device_count()` |


### 코드: test_python_api.py

```python
import rebel

# NPU 개수 확인
count = rebel.device_count()
print(f"NPU count: {count}")

# NPU별 사용 가능 여부 및 이름 확인
for i in range(count):
    available = rebel.npu_is_available(i)

    print(f"NPU {i}")
    print(f"  Available: {available}")

    if available:
        name = rebel.get_npu_name(i)
        print(f"  Name: {name}")
```

### 실행 결과

```bash
NPU count: 1
NPU 0
  Available: True
  Name: RBLN-CA22
```

이 환경에서는 NPU 1개가 확인되며, 인덱스 `0`의 장치 이름은 `RBLN-CA22`이고 사용 가능 여부는 `True`로 반환됩니다.

## 5. ResNet50 컴파일(compile) 및 추론(inference)

[RBLN SDK ResNet50 튜토리얼](https://docs.rbln.ai/v0.11.2/ko/software/api/python/tutorial/basic/pytorch_resnet50.html)을 참조합니다.

### 컴파일

TorchVision에서 사전 학습된 ResNet50 모델을 불러와 추론 모드로 설정한 후, RBLN Python API로 컴파일합니다.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
import rebel

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

compiled_model = rebel.compile_from_torch(
    model,
    [('input', [1, 3, 224, 224], torch.float32)],
)
compiled_model.save('resnet50.rbln')
```

### 컴파일 결과

```bash
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /home/hellonpu/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:00<00:00, 186MB/s]
2026-10-03 10:42:57,585 INFO [rebel-compiler] Export done. Elapsed time: 0:00:00
2026-10-03 10:42:59,378 INFO [rebel-compiler] Exported model conversion done. Elapsed time: 0:00:00, Memory change: 2.57 MB
2026-10-03 10:42:59,426 INFO [rebel-compiler] RBLN SDK compiler version: 0.10.1
2026-10-03 10:42:59,427 INFO [rebel-compiler] -- Target NPU: RBLN-CA22
2026-10-03 10:42:59,427 INFO [rebel-compiler] -- Tensor parallel size: 1
2026-10-03 10:42:59,431 INFO [rebel-compiler] +-------------------------------------------------+
2026-10-03 10:42:59,431 INFO [rebel-compiler] |Compile(#0), mod_name=default, input_info_index=0|
2026-10-03 10:42:59,431 INFO [rebel-compiler] +-------------------------------------------------+
Computation graph generation ████████████████████████████████████████ 100% 00:00
Computation graph optimization  ████████████████████████████████████████ 100% 00:00
2026-10-03 10:43:02,726 INFO [rebel-compiler] Serializing compiled model to resnet50.rbln ...
2026-10-03 10:43:02,797 INFO [rebel-compiler] Compiled model serialized. Elasped time: 0:00:00
```

### 모델 추론

예제 이미지를 다운로드하고 전처리한 후, RBLN Runtime으로 컴파일된 모델을 로드하여 추론을 실행하고 최상위 예측 클래스를 출력합니다.

```python
import urllib.request
from torchvision.io.image import read_image
from rebel import Runtime

weights = ResNet50_Weights.DEFAULT

urllib.request.urlretrieve('https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg', 'input.jpg')
img = read_image('input.jpg')
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

module = Runtime('resnet50.rbln', tensor_type='pt')
output = module(batch)
_, idx = torch.topk(output, 1, dim=1)
pred_class = weights.meta['categories'][idx]
print('Top-1 Predicted Class:', pred_class)
```

### 모델 추론 결과

```bash
2026-10-03 10:46:44,347 INFO [rebel-compiler] Load model completed. Elasped time: 0:00:00
Top-1 Predicted Class: tabby
```

## 6. Llama3.2-1B eager mode 실행

[RBLN SDK Llama 튜토리얼](https://docs.rbln.ai/v0.11.2/ko/software/rbln_pytorch/tutorial_llama.html)을 참조합니다.

### 실행
연산 장치 지정을 cuda 또는 cpu 대신 rbln으로 변경하는 부분을 제외하면, GPU 또는 CPU 환경에서 사용하는 코드와 동일한 구성입니다.

```python
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
device = "rbln"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None,
)
model.to(device)

prompt = "What is the capital of Korea?"
inputs = tokenizer(prompt, return_tensors="pt")

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=64,
    num_return_sequences=1,
    do_sample=False,
    top_p=None,
    temperature=None,
)

prompt_length_tokens = input_ids.shape[1]
generated_text = tokenizer.decode(
    outputs[0][prompt_length_tokens:], skip_special_tokens=True
).strip()
generated_text = re.sub(r"\[duplicate\]\n?", "", generated_text)

print(f"Q: {prompt}")
print(f"A: {generated_text}")
```

### 실행 결과

```text
Q: What is the capital of Korea?
A: Seoul
Seoul is the capital and largest city of South Korea. It is located in the southeastern part of the country and is known for its vibrant culture, rich history, and modern architecture. Seoul is home to many famous landmarks, including the Gyeongbokgung Palace, the Bukchon Hanok
```