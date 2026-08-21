"""embodiment 스키마를 YAML 에서 읽는다.

RoboMIND 2.0 은 embodiment 12 종이고, 서로 다른 것은 카메라 이름·개수, arm 과
end-effector 의 폭, 추가 스트림(chassis / tactile / head), instruction 이 어디
있는가뿐이다. 그것들이 전부 데이터라서 이 파일에는 embodiment 이름이 없다.

검증이 엄격한 이유: `ur` 과 `ur_dex` 는 스트림 이름이 완전히 같고
`end_effector_*_position` 의 폭만 1 과 12 로 다르다. 오타나 누락이 조용히 통과하면
12-DoF 손을 gripper 로 읽고도 에러가 나지 않는다.
"""

from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

INSTRUCTION_SOURCES = frozenset({"zh_file", "h5_metadata", "dirname"})
LAYOUTS = frozenset({"real", "sim"})

_TOP_LEVEL = frozenset(
    {"robot_type", "cameras", "streams", "extra", "instruction", "layout", "fps"}
)
_CAMERA = frozenset({"depth"})
_STREAM = frozenset({"width"})
_EXTRA = frozenset({"shape"})
_INSTRUCTION = frozenset({"source"})


class ConfigError(ValueError):
    """Raised for a malformed embodiment config."""


@dataclass(frozen=True)
class Camera:
    name: str
    depth: bool


@dataclass(frozen=True)
class Stream:
    name: str
    width: int


@dataclass(frozen=True)
class Extra:
    """A per-frame array that is neither a camera nor a puppet/master stream.

    ``agilex_mobile`` is the only embodiment with one today: two tactile pads at
    ``tactile_observations/tactile_{left,right}_align/data``, shape ``(T, 2, 6)``.
    """

    name: str
    group: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class EmbodimentConfig:
    embodiment: str
    robot_type: str
    cameras: tuple[Camera, ...]
    streams: tuple[Stream, ...]
    extras: tuple[Extra, ...]
    instruction_source: str
    layout: str
    # Only `sim` sets this. A real episode's fps is computed from its own
    # timestamps because it varies from 7 to 101 Hz across embodiments and
    # from episode to episode within one.
    fps: int | None

    def stream(self, name: str) -> Stream | None:
        for stream in self.streams:
            if stream.name == name:
                return stream
        return None


def _reject_unknown(keys, allowed, origin: str, what: str) -> None:
    for key in keys:
        if key not in allowed:
            raise ConfigError(f"{origin}: unknown key {key!r} in {what}")


def _mapping(value, origin: str, what: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{origin}: {what} must be a mapping")
    return value


def parse(raw: dict, embodiment: str, origin: str) -> EmbodimentConfig:
    raw = _mapping(raw, origin, "the config")
    _reject_unknown(raw, _TOP_LEVEL, origin, "the config")

    robot_type = raw.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type:
        raise ConfigError(f"{origin}: robot_type must be a non-empty string")

    cameras = []
    for name, body in _mapping(raw.get("cameras"), origin, "cameras").items():
        body = _mapping(body, origin, f"cameras.{name}")
        _reject_unknown(body, _CAMERA, origin, f"cameras.{name}")
        cameras.append(Camera(name=name, depth=bool(body.get("depth", False))))
    if not cameras:
        raise ConfigError(f"{origin}: cameras must not be empty")

    streams = []
    for name, body in _mapping(raw.get("streams"), origin, "streams").items():
        body = _mapping(body, origin, f"streams.{name}")
        _reject_unknown(body, _STREAM, origin, f"streams.{name}")
        width = body.get("width")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ConfigError(f"{origin}: streams.{name}.width must be >= 1")
        streams.append(Stream(name=name, width=width))
    if not streams:
        raise ConfigError(f"{origin}: streams must not be empty")

    extras = []
    for group, members in _mapping(raw.get("extra"), origin, "extra").items():
        for name, body in _mapping(members, origin, f"extra.{group}").items():
            body = _mapping(body, origin, f"extra.{group}.{name}")
            _reject_unknown(body, _EXTRA, origin, f"extra.{group}.{name}")
            shape = body.get("shape")
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
                for dim in shape
            ):
                raise ConfigError(
                    f"{origin}: extra.{group}.{name}.shape must be positive ints"
                )
            extras.append(Extra(name=name, group=group, shape=tuple(shape)))

    instruction = _mapping(raw.get("instruction"), origin, "instruction")
    _reject_unknown(instruction, _INSTRUCTION, origin, "instruction")
    source = instruction.get("source")
    if source not in INSTRUCTION_SOURCES:
        raise ConfigError(
            f"{origin}: instruction.source must be one of "
            f"{', '.join(sorted(INSTRUCTION_SOURCES))}, got {source!r}"
        )

    layout = raw.get("layout")
    if layout not in LAYOUTS:
        raise ConfigError(
            f"{origin}: layout must be one of {', '.join(sorted(LAYOUTS))}, "
            f"got {layout!r}"
        )

    fps = raw.get("fps")
    if layout == "sim":
        if not isinstance(fps, int) or isinstance(fps, bool) or fps < 1:
            raise ConfigError(
                f"{origin}: layout: sim requires fps (a positive int): a simulated "
                "episode's timestamps do not advance, so fps cannot be measured"
            )
    elif fps is not None:
        raise ConfigError(
            f"{origin}: layout: real must not set fps: it is computed per episode "
            "from camera_observations/timestamp, and varies from 7 to 101 Hz"
        )

    return EmbodimentConfig(
        embodiment=embodiment,
        robot_type=robot_type,
        cameras=tuple(cameras),
        streams=tuple(streams),
        extras=tuple(extras),
        instruction_source=source,
        layout=layout,
        fps=fps,
    )


def available() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


def load(name: str) -> EmbodimentConfig:
    import yaml

    path = CONFIG_DIR / f"{name}.yaml"
    if Path(name).name != name or not path.is_file():
        raise ConfigError(
            f"unknown embodiment {name!r}. available: {', '.join(available())}"
        )
    return parse(yaml.safe_load(path.read_text()), embodiment=name, origin=str(path))


def load_all() -> list[EmbodimentConfig]:
    return [load(name) for name in available()]
