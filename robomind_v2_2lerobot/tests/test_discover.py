"""트리 탐색. embodiment 는 경로에서 읽고, 플래그로 받지 않는다.

V2.0 은 `data/<embodiment>/...` 로 시작하므로 어떤 config 를 쓸지 컨버터가 스스로
안다. v1 은 `--embodiments` 를 손으로 받았고, 이름이 틀리면 조용히 0 개를 만들었다.
"""

from fixtures import write_episode

from robomind_v2_utils.reader import discover


def test_finds_a_real_episode(tmp_path):
    write_episode(tmp_path, "tienyi", "flip_white_cup_upright", "1211_100948")

    (ref,) = list(discover([tmp_path]))

    assert ref.embodiment == "tienyi"
    assert ref.task == "flip_white_cup_upright"
    assert ref.path.name == "trajectory.hdf5"


def test_finds_a_sim_episode_named_after_its_id(tmp_path):
    """sim 의 파일명은 trajectory.hdf5 가 아니라 <episode_id>.hdf5 다."""
    write_episode(
        tmp_path, "franka_sim", "103-place_cup_on_holder", "4103000-2025_09_01", layout="sim"
    )

    (ref,) = list(discover([tmp_path]))

    assert ref.embodiment == "franka_sim"
    assert ref.path.name == "4103000-2025_09_01.hdf5"


def test_sim_task_keeps_its_numeric_prefix(tmp_path):
    """task 이름은 디렉토리 이름 그대로다 — instruction 으로 쓸 수 있어야 한다."""
    write_episode(
        tmp_path, "franka_sim", "103-place_cup_on_holder", "4103000-2025_09_01", layout="sim"
    )

    (ref,) = list(discover([tmp_path]))

    assert ref.task == "103-place_cup_on_holder"


def test_five_source_roots_collapse_onto_one_embodiment(tmp_path):
    """Franka 는 repo 5 개인데 embodiment 는 하나다 — 전부 data/franka 로 시작한다."""
    roots = []
    for part in range(1, 6):
        root = tmp_path / f"part_{part}"
        write_episode(root, "franka", f"task_{part}", "0509_130459")
        roots.append(root)

    refs = list(discover(roots))

    assert len(refs) == 5
    assert {ref.embodiment for ref in refs} == {"franka"}
    assert {ref.task for ref in refs} == {f"task_{part}" for part in range(1, 6)}


def test_results_are_sorted_and_stable(tmp_path):
    for stamp in ("0523_144049", "0523_143924", "0523_144014"):
        write_episode(tmp_path, "agilex", "close_fridge_door", stamp)

    paths = [ref.path for ref in discover([tmp_path])]

    assert paths == sorted(paths)


def test_a_task_with_no_episodes_yields_nothing(tmp_path):
    """ark_mobile/grab_beaker_from_left_and_place_on_right 가 실제로 이 모양이다:
    zh_description.txt 만 있고 에피소드가 0 개다."""
    task = tmp_path / "data" / "ark_mobile" / "grab_beaker_from_left_and_place_on_right"
    task.mkdir(parents=True)
    (task / "zh_description.txt").write_text("취해")

    assert list(discover([tmp_path])) == []


def test_missing_data_directory_yields_nothing(tmp_path):
    assert list(discover([tmp_path])) == []
