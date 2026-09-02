# 14. 용어집

표기를 통일하기 위한 용어 목록입니다. 새 용어가 생기면 여기에 추가합니다.

| 용어 | 표기 | 설명 |
|---|---|---|
| NPU | NPU | Neural Processing Unit. AI 연산에 특화된 프로세서 |
| GPU | GPU | Graphics Processing Unit. 범용 병렬 연산 가속기 |
| 추론 | 추론 (inference) | 학습된 모델로 결과를 내는 과정 |
| 양자화 | 양자화 (quantization) | 가중치·활성값을 낮은 정밀도로 표현해 연산·메모리를 줄이는 기법 |
| 컴파일 | 컴파일 (compile) | 모델을 특정 하드웨어에서 실행 가능한 형태로 변환 |
| 지연 시간 | latency | 요청 하나를 처리하는 데 걸리는 시간 |
| 처리량 | throughput | 단위 시간당 처리한 요청/토큰 수 |
| 리벨리온 | 리벨리온 (Rebellions) | NPU를 만드는 회사. 제품군 ATOM, REBEL |
| RBLN SDK | RBLN SDK | 리벨리온 NPU용 개발 도구 모음 |
| KV 캐시 | KV 캐시 (KV cache) | LLM 추론에서 이전 토큰의 키·값을 저장하는 메모리 |
| vLLM | vLLM | 고성능 LLM 추론·서빙 엔진 |
| PE | PE (Processing Element) | NPU에서 곱셈과 누산(MAC)을 수행하는 계산 단위 |
| MAC | MAC (Multiply-Accumulate) | 곱한 뒤 누적하는 신경망의 기본 연산 |
| 온칩 / 오프칩 | 온칩 (on-chip) / 오프칩 (off-chip) | 각각 칩 안에 있어 빠른 영역과 칩 바깥에 있어 느린 영역 |
| SRAM / DRAM | SRAM / DRAM | 각각 6T 셀을 쓰는 빠르고 비싼 메모리(온칩 버퍼용)와 1T1C 셀을 쓰는 크고 저렴한 메모리(오프칩 대용량) |
| HBM / GDDR / LPDDR | HBM / GDDR / LPDDR | DRAM의 종류. HBM은 DRAM을 수직 적층해 초광폭으로 연결한 고대역폭 메모리 |
| 메모리 계층 | 메모리 계층 (memory hierarchy) | 레지스터·스크래치패드·글로벌 버퍼·DRAM처럼 가까울수록 작고 빠른 저장소를 층층이 둔 구조 |
| 스크래치패드 | 스크래치패드 (scratchpad) | 컴파일러·소프트웨어가 직접 관리하는 온칩 메모리. NPU 버퍼의 일반적인 형태 |
| 캐시 | 캐시 (cache) | 하드웨어가 자동으로 관리하는 온칩 메모리. 주로 CPU·GPU에서 사용 |
| 글로벌 버퍼 | 글로벌 버퍼 (global buffer) | PE 배열이 공유하는 온칩 SRAM. 리벨리온 ATOM의 Shared SRAM에 해당 |
| DMA / NoC | DMA / NoC | 각각 메모리 계층 사이에서 데이터를 옮기는 엔진과 칩 내부에서 데이터를 이동하는 네트워크 |
| 대역폭 | 대역폭 (bandwidth) | 1초 동안 옮길 수 있는 데이터 양 |
| 메모리 벽 | 메모리 벽 (memory wall) | 메모리 성능이 연산 성능의 향상 속도를 따라가지 못해 누적된 병목 |
| 연산 강도 | 연산 강도 (arithmetic intensity) | 정한 메모리 경계를 통과한 1바이트당 연산 수. 단위 OP/B 또는 FLOP/B |
| Roofline | Roofline 모델 | 최대 연산 성능과 `메모리 대역폭 × 연산 강도` 중 작은 값을 성능 상한으로 보고 병목을 판단하는 모델 |
| 메모리 바운드 / 연산 바운드 | 메모리 바운드 (memory-bound) / 연산 바운드 (compute-bound) | 각각 데이터 공급이 성능을 제한하는 상태와 연산기가 성능을 제한하는 상태 |
| 타일링 | 타일링 (tiling) | 큰 데이터를 온칩 메모리에 들어갈 크기의 조각으로 나누어 처리하는 루프 변환 |
| GEMM / GEMV | GEMM / GEMV | 각각 행렬×행렬 곱셈과 행렬×벡터 곱셈 |
| 데이터플로우 | 데이터플로우 (dataflow) | 어떤 데이터를 PE에 고정하고 어떤 데이터를 이동시켜 재사용할지 정하는 설계 전략. Weight/Output/Input Stationary 등 |
| 부분합 | 부분합 (partial sum) | 출력 하나를 완성하기 위해 값을 계속 더해 나가는 중간 결과 |
| 시스톨릭 어레이 | 시스톨릭 어레이 (systolic array) | PE를 격자로 연결하고 데이터와 부분합을 이웃 PE로 규칙적으로 전달하는 구조. TPU v1이 대표 |

> 표기 원칙: 한글 우선, 처음 등장할 때 괄호로 영문 병기. 영문 약어(NPU, GPU, vLLM)는 그대로 사용.
