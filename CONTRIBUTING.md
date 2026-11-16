# 기여 가이드

Hello, NPU 전자책에 함께 글을 쓰는 방법입니다.

## 저장소 구조

```text
ebook/
  TOC.md        # 목차. 위키독스 페이지 구조와 페이지 제목을 정합니다
  README.md     # 책 제목(첫 줄 #)과 책 소개. 위키독스 책 소개에 그대로 보입니다
  pages/        # 본문 (각 장 = 한 파일, 본문에 H1 없음)
  assets/       # 이미지 (장별 폴더: assets/NN-slug/)
  templates/    # 챕터 작성 템플릿 (발행 안 됨)
  AGENTS.md     # 작성 규약 (사람·AI 공통)
  CLAUDE.md     # Claude Code 자동화 안내
  .github/workflows/mirror-wikidocs.yml  # main을 위키독스 연동 미러로 복사
```

`README.md`는 위키독스 책 소개로 그대로 쓰이므로 저장소 안내는 이 문서에 둡니다.

## 발행 경로

1. `main`에 push(머지)되면 GitHub Action(`.github/workflows/mirror-wikidocs.yml`)이 위키독스 연동용 미러 저장소(`ugiugi0823/hello-npu-ebook`)로 `main`을 그대로 복사합니다. Action은 `WIKIDOCS_MIRROR_TOKEN` 시크릿을 씁니다.
2. 위키독스가 미러 저장소를 동기화해 책에 반영합니다.
3. 반영이 안 보이면 위키독스의 책 수정 화면에서 깃허브 메뉴의 "지금 동기화"를 누릅니다.

미러는 한 방향입니다. 미러 저장소나 위키독스에서 직접 고치지 말고, 원고는 항상 이 저장소에서 고칩니다.

## 빠른 시작

1. 저장소를 클론하고 브랜치를 만듭니다: `git switch -c docs/08-serving`
2. `pages/`에서 담당 장을 작성·수정합니다. (형식은 [templates/chapter.md](templates/chapter.md))
3. 새 페이지를 만들었다면 [TOC.md](TOC.md)에 등록합니다.
4. 커밋 후 Pull Request를 올립니다. 리뷰를 거쳐 `main`에 머지되면 미러 Action을 거쳐 위키독스에 발행됩니다(아래 [발행 경로](#발행-경로)).

## 규약

작성 규약은 [AGENTS.md](AGENTS.md)를 따릅니다. 핵심만:

- 한국어 합니다체, 용어는 [용어집](pages/12-glossary.md) 기준
- 파일명 `NN-slug.md`(두 자리 번호). 본문에 H1(`#`)을 쓰지 않고 `## 개요`로 시작합니다. 장 제목은 `TOC.md`에만 쓰고, 파일 번호와 TOC 제목 번호를 맞춥니다.
- 추정값·지어낸 수치 금지. 실제 결과만, 없으면 `> 📝 실습 채움`

## Git이 익숙하지 않다면

마크다운으로 글만 작성해 장 담당자에게 전달하면, 담당자가 PR로 반영합니다. 형식이 조금 어긋나도 괜찮습니다 — `/normalize`로 다듬습니다.

## 자동화 도움

Claude Code 슬래시 명령으로 초안 생성(`/draft`)·교정(`/normalize`)·목차 갱신(`/toc`)·발행(`/publish`)을 할 수 있습니다. ([CLAUDE.md](CLAUDE.md))
