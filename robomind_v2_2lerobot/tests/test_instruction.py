"""instruction 의 출처가 세 갈래고, 어느 쪽에도 없는 경우가 있다.

  zh_file      real 10 종 — task 디렉토리의 zh_description.txt, 중국어
  h5_metadata  sim 2 종  — metadata/language_instruction, 영어 문장
  dirname      ur_dex    — 둘 다 없다

zh 파일이 없는 task 가 실제로 하나 있다(ur/assemble_lego_letters, 521 에피소드).
그때 변환을 멈추지 않고 디렉토리 이름으로 내려간다 — 521 에피소드를 버리는 것보다 낫다.
"""

import h5py
import pytest
import yaml
from fixtures import write_episode

from robomind_v2_utils import configs
from robomind_v2_utils.reader import EpisodeRef, instruction


def probe(tmp_path, source: str, layout: str = "real"):
    body = {
        "robot_type": "probe",
        "cameras": {"camera_top": {"depth": True}},
        "streams": {"arm_left_position": {"width": 7}},
        "instruction": {"source": source},
        "layout": layout,
    }
    if layout == "sim":
        body["fps"] = 30
    (tmp_path / "probe.yaml").write_text(yaml.safe_dump(body))
    original, configs.CONFIG_DIR = configs.CONFIG_DIR, tmp_path
    try:
        return configs.load("probe")
    finally:
        configs.CONFIG_DIR = original


def ask(path, ref, config):
    with h5py.File(path, "r") as handle:
        return instruction(ref, config, handle)


def test_zh_file_is_read_from_the_task_directory(tmp_path):
    config = probe(tmp_path, "zh_file")
    path = write_episode(
        tmp_path, "probe", "fold_clothes", "0001_000000",
        streams={"arm_left_position": 7}, instruction="叠衣服",
    )
    ref = EpisodeRef(embodiment="probe", task="fold_clothes", path=path)

    assert ask(path, ref, config) == "叠衣服"


def test_h5_metadata_is_read_from_inside_the_file(tmp_path):
    config = probe(tmp_path, "h5_metadata", layout="sim")
    path = write_episode(
        tmp_path, "probe", "103-place_cup_on_holder", "0002_000000",
        streams={"arm_left_position": 7}, layout="sim",
        instruction="Place the cup on the holder",
    )
    ref = EpisodeRef(embodiment="probe", task="103-place_cup_on_holder", path=path)

    assert ask(path, ref, config) == "Place the cup on the holder"


def test_dirname_becomes_a_readable_sentence(tmp_path):
    config = probe(tmp_path, "dirname")
    path = write_episode(
        tmp_path, "probe", "box_in_the_basket", "0003_000000",
        streams={"arm_left_position": 7},
    )
    ref = EpisodeRef(embodiment="probe", task="box_in_the_basket", path=path)

    assert ask(path, ref, config) == "box in the basket"


def test_a_missing_zh_file_falls_back_to_the_directory_name(tmp_path):
    """ur/assemble_lego_letters 가 실제로 이 모양이다 — 521 에피소드를 버리지 않는다."""
    config = probe(tmp_path, "zh_file")
    path = write_episode(
        tmp_path, "probe", "assemble_lego_letters", "0004_000000",
        streams={"arm_left_position": 7}, instruction=None,
    )
    ref = EpisodeRef(embodiment="probe", task="assemble_lego_letters", path=path)

    assert ask(path, ref, config) == "assemble lego letters"


def test_an_empty_zh_file_falls_back_too(tmp_path):
    config = probe(tmp_path, "zh_file")
    path = write_episode(
        tmp_path, "probe", "close_laptop", "0005_000000",
        streams={"arm_left_position": 7}, instruction="   ",
    )
    ref = EpisodeRef(embodiment="probe", task="close_laptop", path=path)

    assert ask(path, ref, config) == "close laptop"


def test_a_sim_prefix_is_stripped_from_the_fallback(tmp_path):
    """sim 의 task 디렉토리에는 101- 같은 숫자 접두가 있다. 문장에 넣지 않는다."""
    config = probe(tmp_path, "dirname")
    path = write_episode(
        tmp_path, "probe", "101-pick_cube_into_plate", "0006_000000",
        streams={"arm_left_position": 7},
    )
    ref = EpisodeRef(embodiment="probe", task="101-pick_cube_into_plate", path=path)

    assert ask(path, ref, config) == "pick cube into plate"


def test_the_result_is_never_empty(tmp_path):
    config = probe(tmp_path, "zh_file")
    path = write_episode(
        tmp_path, "probe", "x", "0007_000000", streams={"arm_left_position": 7}
    )
    ref = EpisodeRef(embodiment="probe", task="x", path=path)

    assert ask(path, ref, config).strip() != ""
