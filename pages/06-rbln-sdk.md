# 리벨리온 NPU 사용 예제

리벨리온 SDK가 미리 설치된 Python 가상환경에서 시작한다. 가상환경을 활성화한 뒤 SDK 관련 패키지와 시스템 라이브러리, NPU 장치를 확인하고 간단한 텐서 연산 및 Python 유틸리티 API를 실행한다.

참고 문서:
- [RBLN SDK v0.11.2 사용자 가이드](https://docs.rbln.ai/v0.11.2/ko/index.html)


## 1. 가상환경 활성화

SDK가 설치된 가상환경을 활성화한다.

```bash
source ~/rbln-env/bin/activate
```

이후 명령은 해당 가상환경이 활성화된 상태에서 실행한다.

## 2. 서버의 SDK 및 장치 확인

### 2-1. 환경변수 확인

`rbln` 문자열이 포함된 환경변수를 확인한다.

```bash
env | grep -Ei 'rbln'
```

출력:

```text
VIRTUAL_ENV=/home/hellonpu/rbln-env
VIRTUAL_ENV_PROMPT=(rbln-env)
```

이 출력에서는 활성화된 가상환경의 경로와 프롬프트 설정을 확인할 수 있다. SDK 설치 여부는 아래 패키지와 라이브러리 확인을 통해 별도로 점검한다.

### 2-2. 설치된 관련 패키지 확인

```bash
pip list | grep -Ei 'rbln|rebel|torch'
```

출력:

```text
optimum-rbln                             0.10.1
rebel-compiler                           0.10.1
torch                                    2.9.0+cpu
torch-rbln                               0.1.7
torchaudio                               2.9.0
torchvision                              0.24.0
vllm_rbln                                0.10.1.post2
```

### 2-3. 시스템 라이브러리 확인

시스템의 공유 라이브러리 캐시에서 리벨리온 관련 라이브러리를 확인한다.

```bash
ldconfig -p 2>/dev/null | grep -Ei 'rbln|rebel'
```

출력:

```text
        librbln-thunk.so.3 (libc6,x86-64) => /lib/librbln-thunk.so.3
        librbln-thunk.so (libc6,x86-64) => /lib/librbln-thunk.so
        librbln-ml.so (libc6,x86-64) => /lib/librbln-ml.so
        librbln-ccl.so.3 (libc6,x86-64) => /lib/librbln-ccl.so.3
        librbln-ccl.so (libc6,x86-64) => /lib/librbln-ccl.so
```

### 2-4. NPU 장치 확인

NVIDIA GPU에서 `nvidia-smi`로 장치 상태를 확인하는 것과 유사하게, 리벨리온 NPU에서는 `rbln-smi`를 사용한다.

```bash
rbln-smi
```

출력:

```text
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

출력에서 확인한 장치 상태는 다음과 같다.

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

[공식 PyTorch RBLN 튜토리얼](https://docs.rbln.ai/latest/ko/software/rbln_pytorch/tutorial_running_n_debugging.html)의 `torch.add` 예제를 실행한다.

`device="rbln"`을 지정해 NPU에 FP16 텐서를 생성하고, 두 텐서의 원소별 덧셈을 수행한다. 코드의 `a + b`는 `torch.add`에 해당하는 연산이다.

### 코드: test_add.py

```python
import torch

device = "rbln"

a = torch.tensor(, dtype=torch.float16, device=device) 
b = torch.tensor(, dtype=torch.float16, device=device)

c = a + b
print(c)
```

### 실행

```bash
python3 test_add.py
```

### 실행 결과

```text
tensor([5., 7., 9.], device='rbln:0', dtype=torch.float16)
```

결과 텐서의 값은 `[5., 7., 9.]`이며, 장치는 `rbln:0`, 데이터 타입은 `torch.float16`으로 표시된다.

## 4. Python 유틸리티 API

`rebel` 모듈의 유틸리티 API로 NPU 개수, 특정 장치의 사용 가능 여부, 장치 이름을 확인할 수 있다.

NVIDIA GPU에서 사용하는 PyTorch CUDA API와 기능상 대응되는 API는 다음과 같다.

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

### 실행

```bash
python3 test_python_api.py
```

### 실행 결과

```text
NPU count: 1
NPU 0
  Available: True
  Name: RBLN-CA22
```

이 환경에서는 NPU 1개가 확인되며, 인덱스 `0`의 장치 이름은 `RBLN-CA22`이고 사용 가능 여부는 `True`로 반환된다.