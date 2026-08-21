"""0 개 변환은 실패다.

v1 은 `h5_<embodiment>/` 를 못 찾으면 아무 것도 yield 하지 않고 에러 없이 끝났고,
에피소드가 전부 skip 되면 출력 디렉토리를 rmtree 하고 정상 종료했다. "0 개 변환됨" 이
성공처럼 보였다. 이 테스트가 그 회귀를 막는다.
"""

import re
from pathlib import Path

import pytest
from fixtures import write_episode

from robomind_v2_h5 import NothingConverted, plan_tasks


def test_tasks_are_grouped_by_embodiment_and_task(tmp_path):
    write_episode(tmp_path, "tienyi", "task_a", "0001_000000")
    write_episode(tmp_path, "tienyi", "task_a", "0002_000000")
    write_episode(tmp_path, "tienyi", "task_b", "0003_000000")

    grouped = plan_tasks([tmp_path])

    assert set(grouped) == {("tienyi", "task_a"), ("tienyi", "task_b")}
    assert len(grouped[("tienyi", "task_a")]) == 2


def test_five_roots_group_into_one_embodiment(tmp_path):
    """Franka 5 개 repo — task 는 다르지만 embodiment 는 하나다."""
    roots = []
    for part in range(1, 6):
        root = tmp_path / f"part_{part}"
        write_episode(root, "franka", f"task_{part}", "0509_130459")
        roots.append(root)

    grouped = plan_tasks(roots)

    assert {embodiment for embodiment, _ in grouped} == {"franka"}
    assert len(grouped) == 5


def test_an_empty_source_raises_rather_than_succeeding(tmp_path):
    from robomind_v2_h5 import main

    with pytest.raises(NothingConverted, match="no episodes"):
        main(src_paths=[tmp_path], output_path=tmp_path / "out", debug=True)


def test_an_unknown_embodiment_raises(tmp_path):
    """디렉토리 이름에 대응하는 config 가 없으면 조용히 넘어가지 않는다."""
    from robomind_v2_h5 import main

    write_episode(tmp_path, "no_such_robot", "task", "0001_000000")

    with pytest.raises(NothingConverted, match="no config"):
        main(src_paths=[tmp_path], output_path=tmp_path / "out", debug=True)


def test_short_episodes_alone_raise(tmp_path):
    """모든 에피소드가 min_frames 미달이면 결과가 0 개다 — 실패여야 한다."""
    from robomind_v2_h5 import main

    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)

    with pytest.raises(NothingConverted):
        main(
            src_paths=[tmp_path],
            output_path=tmp_path / "out",
            min_frames=1000,
            debug=True,
        )


def test_no_embodiment_name_appears_in_the_converter():
    """변환 로직에 embodiment 이름이 등장하면 안 된다.

    부분문자열이 아니라 **단어 단위**로 본다. `ark` 는 dark·marks·remarkable 같은
    평범한 영어 단어에 들어 있어서, 부분문자열 검사는 산문에서 오탐을 낸다
    (실제로 "Marks" 가 걸린 적이 있다). 브랜드 표기도 같이 막는다 — 한 번은
    소문자 슬러그만 보다가 "UR5"/"AgileX" 가 통과했다.
    """
    slugs = [
        "agilex", "agilex_mobile", "ark", "ark_mobile", "franka", "franka_sim",
        "tienkung", "tienkung_sim", "tienyi", "tienyi_mobile", "ur_dex",
    ]
    brands = ["UR5", "AgileX", "Agilex", "Franka", "Tianyi", "ARX", "TienKung"]
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(n) for n in slugs + brands) + r")\b"
    )

    root = Path(__file__).resolve().parents[1]
    sources = [root / "robomind_v2_h5.py", *(root / "robomind_v2_utils").glob("*.py")]

    for path in sources:
        found = sorted(set(pattern.findall(path.read_text())))
        assert found == [], f"{path.name} names embodiments: {found}"
