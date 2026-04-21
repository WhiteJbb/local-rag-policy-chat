"""
clean_md.py
마크다운 파일의 깨진 테이블, 중복 섹션, 페이지 번호 등을 정제합니다.
사용법: python clean_md.py "docs/회사 운영규정.md"
"""

import re
import sys
from pathlib import Path


def is_broken_table_row(line: str) -> bool:
    """의미 없는 테이블 구분선 또는 빈 셀로만 구성된 행"""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    # | --- | --- | 형태
    if re.fullmatch(r"(\|\s*-+\s*)+\|?", stripped):
        return True
    # 셀이 전부 공백인 행: |  |  |  |
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if all(c == "" for c in cells):
        return True
    return False


def extract_table_text(line: str) -> str:
    """테이블 행에서 의미있는 텍스트만 추출"""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    meaningful = [c for c in cells if c and c != "---" and not re.fullmatch(r"-+", c)]
    return "  ".join(meaningful)


def is_standalone_page_number(line: str) -> bool:
    """단독 페이지 번호 행 (숫자만 있는 줄)"""
    stripped = line.strip()
    return bool(re.fullmatch(r"\d{1,3}", stripped))


def is_noise_line(line: str) -> bool:
    """의미 없는 단일 문자/기호 행"""
    stripped = line.strip()
    if not stripped:
        return True
    # 단일 특수문자/기호
    if stripped in {"~", "or", "/", "],", "],", ".", "0 5"}:
        return True
    # 섹션 번호만 있는 줄 (01~07)
    if re.fullmatch(r"0[1-9]", stripped):
        return True
    # 시간 범위 표기 (~22, ~09, 06/ 등)
    if re.fullmatch(r"~\d{2}|or|\d{2}/", stripped):
        return True
    # 한 글자 (각, 의, 등)
    if len(stripped) == 1:
        return True
    return False


def clean_html(line: str) -> str:
    """<br> 태그를 공백으로 치환, 기타 HTML 제거, ü → - 변환"""
    line = re.sub(r"<br\s*/?>", " ", line)
    line = re.sub(r"<[^>]+>", "", line)
    # PDF 추출 시 체크마크/불릿이 깨진 문자 → 리스트 기호로 변환
    line = re.sub(r"ü\s*", "- ", line)
    return line


def clean_line(line: str) -> str | None:
    """한 줄을 정제. None 반환 시 해당 줄 제거."""
    # 단독 페이지 번호
    if is_standalone_page_number(line):
        return None
    # HTML 태그 제거
    line = clean_html(line)
    # 노이즈 줄
    if is_noise_line(line):
        return None
    # 깨진 테이블 구분선
    if is_broken_table_row(line):
        return None
    # 테이블 행 → 텍스트로 변환
    if line.strip().startswith("|"):
        text = extract_table_text(line)
        return text if text else None
    return line.rstrip()


def remove_duplicate_sections(lines: list[str]) -> list[str]:
    """연속된 중복 헤더 블록 제거"""
    result = []
    seen_blocks: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        # 헤더 시작 감지
        if line.startswith("#"):
            # 다음 헤더까지의 블록을 키로 사용
            block_lines = [line]
            j = i + 1
            while j < len(lines) and not lines[j].startswith("#"):
                block_lines.append(lines[j])
                j += 1
            block_key = "\n".join(block_lines).strip()
            if block_key in seen_blocks:
                i = j
                continue
            seen_blocks.add(block_key)
            result.extend(block_lines)
            i = j
        else:
            result.append(line)
            i += 1
    return result


def collapse_blank_lines(lines: list[str]) -> list[str]:
    """연속 빈 줄을 최대 1개로 압축"""
    result = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank
    return result


def clean_file(input_path: str) -> None:
    src = Path(input_path)
    dst = src.with_stem(src.stem + "_clean")

    raw_lines = src.read_text(encoding="utf-8").splitlines()
    print(f"원본 줄 수: {len(raw_lines)}")

    # 1단계: 줄별 정제
    cleaned: list[str] = []
    for line in raw_lines:
        result = clean_line(line)
        if result is not None:
            cleaned.append(result)

    # 2단계: 중복 섹션 제거
    cleaned = remove_duplicate_sections(cleaned)

    # 3단계: 연속 빈 줄 압축
    cleaned = collapse_blank_lines(cleaned)

    dst.write_text("\n".join(cleaned), encoding="utf-8")
    print(f"정제 완료: {len(cleaned)}줄 → {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python clean_md.py <파일경로>")
        sys.exit(1)
    clean_file(sys.argv[1])
