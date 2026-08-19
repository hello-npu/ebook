# 기여 가이드

Hello, NPU 전자책에 함께 글을 쓰는 방법입니다.

## 빠른 시작

1. 저장소를 클론하고 브랜치를 만듭니다: `git switch -c docs/08-serving`
2. `pages/`에서 담당 장을 작성·수정합니다. (형식은 [templates/chapter.md](templates/chapter.md))
3. 새 페이지를 만들었다면 [TOC.md](TOC.md)에 등록합니다.
4. 커밋 후 Pull Request를 올립니다. 리뷰를 거쳐 머지하면, `main` 반영 시 위키독스에 자동 발행됩니다.

## 규약

작성 규약은 [AGENTS.md](AGENTS.md)를 따릅니다. 핵심만:

- 한국어 합니다체, 용어는 [용어집](pages/13-glossary.md) 기준
- 파일명 `NN-slug.md`(두 자리 번호), H1 번호와 파일 번호 일치
- 추정값·지어낸 수치 금지. 실제 결과만, 없으면 `> 📝 실습 채움`

## Git이 익숙하지 않다면

마크다운으로 글만 작성해 장 담당자에게 전달하면, 담당자가 PR로 반영합니다. 형식이 조금 어긋나도 괜찮습니다 — `/normalize`로 다듬습니다.

## 자동화 도움

Claude Code 슬래시 명령으로 초안 생성(`/draft`)·교정(`/normalize`)·목차 갱신(`/toc`)·발행(`/publish`)을 할 수 있습니다. ([CLAUDE.md](CLAUDE.md))
