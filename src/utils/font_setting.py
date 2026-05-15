import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 우선순위 순으로 시도할 한글 폰트 목록
_KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",      # Windows 기본
    "NanumGothic",        # 나눔고딕
    "NanumSquare Neo variable",
    "NanumSquareOTF",
    "Noto Sans KR",       # 구글 노토 한글
    "AppleGothic",        # macOS
    "Gulim",              # 굴림
    "Batang",             # 바탕
]

def _find_korean_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _KOREAN_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    raise RuntimeError(
        f"한글 폰트를 찾을 수 없습니다. "
        f"설치된 폰트 목록: {sorted(available)}"
    )

def apply() -> str:
    """
    matplotlib에 한글 폰트를 적용한다.
    선택된 폰트 이름을 반환한다.
    사용법: from src.utils.font_setting import apply; apply()
    """
    font_name = _find_korean_font()
    matplotlib.rc("font", family=font_name)
    matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
    return font_name


# 이 파일을 직접 실행하면 선택된 폰트를 출력
if __name__ == "__main__":
    name = apply()
    print(f"한글 폰트 적용됨: {name}")
