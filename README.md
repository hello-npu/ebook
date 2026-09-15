# 리벨리온 NPU 실전 가이드

Hello, NPU 랩이 6개월 동안 리벨리온 서버용 NPU에 실제 AI 모델을 올려보며 정리하는 한국어 실전 가이드입니다. 교과서가 아니라, 직접 돌려보고 막히고 해결한 기록을 모읍니다.

이 저장소는 [위키독스](https://wikidocs.net)와 연동되어, `main` 브랜치에 push하면 자동으로 전자책에 반영됩니다.

## 다루는 내용

| 장 | 내용 |
|---|---|
| 01~03 | NPU 개념, GPU 비교, RBLN SDK 환경 셋업 |
| 04~07 | 모델 올리기·컴파일·최적화, 비전/NLP/생명과학 모델 구동 |
| 08~10 | vLLM 이해, NPU에 LLM 서빙, 서빙 문제점 개선 |
| 11~13 | 벤치마크 모음, 트러블슈팅, 용어집 |

전체 목차는 [TOC.md](TOC.md)를 참고하세요.

## 저장소 구조

```
ebook/
  TOC.md        # 목차 (위키독스 페이지 구조 정의)
  pages/        # 본문 (각 장 = 한 파일)
  assets/       # 이미지
  templates/    # 챕터 작성 템플릿
  AGENTS.md     # 작성 규약 (사람·AI 공통)
  CLAUDE.md     # Claude Code 자동화 안내
  CONTRIBUTING.md
```

## 함께 쓰는 법

1. 새 글은 [챕터 템플릿](templates/chapter.md) 형식으로 `pages/`에 작성합니다.
2. 용어 표기는 [용어집](pages/13-glossary.md) 기준으로 통일합니다.
3. 브랜치를 만들고 Pull Request로 올린 뒤 리뷰를 거쳐 머지합니다.
4. 작성·정리·발행 자동화는 [CLAUDE.md](CLAUDE.md)의 슬래시 명령을 사용합니다.

자세한 규약은 [AGENTS.md](AGENTS.md)와 [CONTRIBUTING.md](CONTRIBUTING.md)를 따릅니다.

> 이 책은 계속 자라는 살아있는 문서입니다. 완성보다 꾸준한 누적을 목표로 합니다.
