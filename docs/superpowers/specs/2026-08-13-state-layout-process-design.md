# 전처리 규약의 프로세스화 설계

작성일: 2026-08-13

## 배경

`/data/taeyoung/data/vla_pretrain_dataset` 아래의 RLDX-1 사전학습 데이터셋을 만든 코드는 존재하지
않는다. ALIN Lab이 2026-07-29에 Gaia/SKT에서 임시로 처리했고 보관하지 않았다고 확인했다
([Notion](https://app.notion.com/p/3ac6cbdff6f680efaa98d85739443184), Q2).

지금까지 한 일은 **역산**이었다. 납품본에서 44슬롯 배치를 복구하고
(`docs/vla-pretrain-state-action-layout.md`), 그 결과를 `dataset_registry`에 스펙으로 적고,
ActionNet 컨버터를 그 스펙에 맞췄다.

이번에 할 일은 그 결과를 **프로세스로 만드는 것**이다. 역산으로 알아낸 것이 지금은 특정
데이터셋을 상정한 코드와 손으로 복사한 값에 흩어져 있다.

- `actionnet2lerobot/`(622줄)은 통째로 ActionNet 전용이다. 데이터셋을 하나 추가할 때마다 패키지를
  하나 더 만드는 구조다.
- `actionnet_utils/config.py`의 `PERMUTATION` — 44슬롯 조립이 ActionNet 컨버터 안에만 있다.
- `dataset_registry/datasets/*.yaml`의 `slots: [0, 7]` — 파생값인데 손으로 적혀 있고, 그래서
  `action_net` / `agibot_dexhand` / `agibot_gripper` / `neural_robocurate` 네 파일에 같은
  블록 8개가 복사되어 있다.

## 요구사항의 출처

이 설계를 결정한 네 가지 답:

1. **재구축본은 당분간 기존 PT 체크포인트와 호환되어야 하고, 이후 canonical 레이아웃으로
   전환한다.** → 레이아웃이 코드가 아니라 교체 가능한 config여야 한다.
2. **재구축의 입력은 upstream 원본이다.** 납품본은 검증 기준으로만 쓴다.
3. **처리 방식(state 규칙 + video 처리)이 config 한 장에 나열되면 좋겠다.**
4. **컨버터가 특정 데이터셋을 상정하지 않으면 좋겠다. 특정 데이터셋 정보는 결국 나중에 추가될
   데이터에 불과하다.**

체크포인트 호환이 왜 슬롯 순서를 강제하는지는 `docs/vla-pretrain-state-action-layout.md`의
"No cross-dataset alignment is needed" 절에 있다. 요약하면, 임베디먼트마다 별도 가중치
행렬 `[36, 64, H]`를 쓰므로 데이터셋 간 정렬은 필요 없지만, **한 임베디먼트 안에서는 학습 때
순서와 추론 때 순서가 같아야 한다.**

## 지도 원칙

> **코드는 이름 붙은 메커니즘의 닫힌 집합을 제공하고, yaml이 고르고 파라미터를 준다.**

이 레포에 이미 두 번 나온 패턴이다 — `registry.register_step`(스텝을 이름으로만 참조),
`configs/encoding/*.yaml`(인코딩 프로파일). 이번에 소스 포맷과 시계 동기화, 블록 순서에도
같은 패턴을 적용한다.

따라서 코드는 **데이터셋 수(35+)가 아니라 메커니즘 수(포맷 4~5개, 동기화 2~3개)에 비례해서**
늘어난다. 데이터셋 추가는 yaml 추가다.

## 목표

- 데이터셋 하나를 추가하는 일이 **yaml 한 장 추가**가 되게 한다. 코드 변경 없이.
- state/action 배치 규약을 공용 step + config로 만든다.
- 슬롯 번호를 config에서 없앤다. 블록 폭과 순서에서 계산되게 한다.
- 규약 전환(호환본 → canonical)이 **파일 한 개 교체**로 끝나게 한다.
- "이 데이터셋을 어떻게 처리하는가"가 **한 파일에 나열**되게 한다.
- 영상 재인코딩 없이 배치만 다시 만들 수 있게 한다.
- 재구축이 **불가능한 데이터셋을 조용히 통과시키지 않는다.**

## 비목표

- `agibot` / `openx` / `robocasa` / `robomind` / `libero` 컨버터 마이그레이션. 기존 컨버터는
  그대로 두고 `_CONVERTERS`에 남긴다. 하나씩 옮길 수 있다.
- `hdf5_episodes` 외의 포맷 리더(RLDS 등). 인터페이스만 열어둔다.
- 프레임 수를 바꾸는 step. 계속 예약 상태.
- OXE 27개 등 나머지 데이터셋 스펙 채우기. 별건으로 일괄 진행하기로 했다.
- S3 업로드. 파이프라인 코어는 디렉토리 → 디렉토리만 다룬다.
- 재구축본 ↔ 납품본 등가성 검증기. 검증할 실데이터는 나중에 들어온다. 그때는 기존
  `dataset_registry/verify.py`가 그대로 최종 확인을 맡는다.

## 세 레이어

| 레이어 | 위치 | 담는 것 | 언제 바뀌나 |
|---|---|---|---|
| **사실** | `dataset_registry/datasets/*.yaml` | upstream sha, 라이선스, 소스 파일 배치, 블록 폭·출처, evidence | 데이터셋 추가 시 |
| **규약** | `dataset_registry/layouts/*.yaml`<br>`lerobot_pipeline/configs/profiles/*.yaml`<br>`lerobot_pipeline/configs/encoding/*.yaml` | 블록 순서, 리사이즈, 인코딩, 출력 버전 | 규약 전환 시 |
| **실행** | `lerobot_pipeline/configs/*.yaml` | 입출력 경로, 워커 수, 오버라이드 | 매 실행 |

경계 규칙: **사실은 "이 데이터가 무엇이고 어디에 어떻게 놓여 있는지", 규약은 "그걸 어떤 순서로
어떻게 굽는지", 실행은 "이번에 어디서 읽어 어디에 쓰는지".**

---

# 입력부 — 소스 서술

## 무엇이 데이터이고 무엇이 코드인가

`actionnet2lerobot` 622줄을 분류한 결과다.

| 남는 것 | 현재 위치 | 성격 | 갈 곳 |
|---|---|---|---|
| `ROBOT_JOINTS=32`, `HAND_JOINTS=12` | config.py | 데이터 | 블록 폭에서 파생 |
| `SOURCE_CAMERA_DIR`, `CAMERA_NAME` | config.py | 데이터 | yaml `video.cameras` (이미 있음) |
| `FPS`, `ROBOT_TYPE`, `RGB_SHAPE` | config.py | 데이터 | yaml (이미 있음) |
| hdf5 키 `state/robot` … | actionnet_utils.py | 데이터 | yaml `source_features` (이미 있음) |
| 파일 배치 `<id>.hdf5`, `<id>/<cam>/rgb.mp4` | actionnet_utils.py | 데이터 | yaml `source.paths` (신규) |
| `metadata.json`의 `id`/`prompt` | actionnet_utils.py | 데이터 | yaml `source.tasks` (신규) |
| `TIMESTAMP_FORMAT` | actionnet_utils.py | 데이터 | yaml `source.clock` (신규) |
| `match_timestamps` + 두 필터 | actionnet_utils.py | **알고리즘** | 이름 붙은 clock 전략 (코드) |
| hdf5 열기/배열 검사/BaseAdapter 배선 | 전반 | **포맷 코드** | `hdf5_episodes` 리더 (코드) |

`config.py` 141줄은 레이아웃을 빼면 전부 레지스트리 yaml과 중복이라 통째로 사라진다.
진짜로 코드여야 하는 것은 `match_timestamps`와 두 필터, 약 40줄이다.

## yaml 서술

```yaml
# dataset_registry/datasets/action_net.yaml
source:
  format: hdf5_episodes
  # 에피소드 id는 이 패턴으로 발견한다
  discover: "*.hdf5"
  paths:
    episode: "{id}.hdf5"
    video: "{id}/{camera}/rgb.mp4"
  tasks:
    file: metadata.json
    key: id
    prompt: prompt
  clock:
    strategy: nearest_timestamp_dedup
    data: timestamp                        # hdf5 안의 시계
    image: "{id}/{camera}/timestamps.json" # 영상 프레임 시계
    image_format: "%Y-%m-%dT%H-%M-%S_%f"
```

`{camera}`는 `video.cameras`의 소스 디렉토리 이름(`top`)으로 치환된다.

## 포맷 리더

`register_format`으로 이름 등록하는 닫힌 집합. 이번에는 하나만 구현한다.

| 이름 | 대상 | 상태 |
|---|---|---|
| `hdf5_episodes` | 에피소드당 hdf5 + 영상 디렉토리 (ActionNet, agibot) | 이번에 구현 |
| `rlds` | tfrecord (OXE 27개) | 인터페이스만 |
| `lerobot` | 이미 LeRobot인 소스 (humanoid_everyday, galaxea) | 인터페이스만 |

**솔직한 한계.** RLDS는 yaml로 서술할 수 없다. 포맷 리더가 코드로 남는 이유이고, 이 집합은
데이터셋 수가 아니라 포맷 수에 비례한다.

`lerobot` 포맷의 소스는 애초에 컨버터가 필요 없다 — 파이프라인이 `state_layout`부터 시작한다.
인터페이스만 열어두는 것은 나중에 소스 피처 이름이 다를 때 매핑할 자리로 쓰기 위해서다.

## 시계 전략

`register_clock`으로 등록. 소스의 로봇 시계와 카메라 시계를 맞추는 방법이다.

| 이름 | 하는 일 | 상태 |
|---|---|---|
| `nearest_timestamp_dedup` | 각 영상 프레임에 가장 가까운 로봇 샘플. 한 샘플은 한 프레임만 갖고, 겹치면 다음 것, 그것도 겹치면 프레임을 버린다. 추가로 시각이 증가하지 않는 샘플과 마지막 로봇 시각 이후의 프레임을 거른다. | 이번에 구현 |
| `index` | 인덱스로 그대로 맞춤 (시계가 하나인 소스) | 이번에 구현 (자명) |

`nearest_timestamp_dedup`은 `FFTAI/fourier-lerobot`의 `convert_hdf5_to_lerobot.py`를 그대로
옮긴 것이다. 상류와 다르게 갈 이유가 없다는 판단을 이미 내렸고, 그 판단이 코드가 아니라
전략 이름으로 남는다.

## 스펙 기반 컨버터

`spec2lerobot/` — 레포의 `<source>2lerobot` 명명을 따르되, 소스가 데이터셋이 아니라 스펙이다.

```bash
python -m spec2lerobot --dataset action_net --src-path /scratch/data \
    --output-path /scratch/out --executor local
```

`SpecAdapter(BaseAdapter)` 하나가 모든 데이터셋을 처리한다. 데이터셋 이름이 코드에 등장하지
않는다. `generic_converter`의 datatrove 배선(`load_tasks` / `load_subset` / `create_dataset` /
`save_episode`)과 PR #1에서 넣은 `prerendered_video`를 그대로 쓴다.

산출물은 **소스 피처뿐**이다. `observation.state` / `action` / `modality.json`은 만들지 않는다.
그것은 `state_layout`의 일이다.

`generic_converter/`는 상류 코드이므로 손대지 않는다.

---

# 출력부 — state 배치

## 슬롯 번호를 없앤다

현재:

```yaml
blocks:
  - {name: left_arm,  slots: [0, 7],  source: {feature: robot, columns: [18, 25]}}
  - {name: left_hand, slots: [7, 13], source: {feature: hand,  columns: [0, 6]}}
```

변경 후 — 데이터셋은 폭과 출처만 말한다:

```yaml
state:
  width: 44                # 계산 결과와 일치해야 하는 검산값
  layout: gr1_body_parts
  source_features:
    robot: {state: state/robot, action: action/robot}
    hand:  {state: state/hand,  action: action/hand}
  blocks:
    left_arm:   {width: 7, source: {feature: robot, columns: [18, 25]}, evidence: measured}
    left_hand:  {width: 6, source: {feature: hand,  columns: [0, 6]},   evidence: measured}
    left_leg:   {width: 6, source: {feature: robot, columns: [0, 6]},   evidence: constant}
    neck:       {width: 3, source: {feature: robot, columns: [15, 18]}, evidence: measured}
    right_arm:  {width: 7, source: {feature: robot, columns: [25, 32]}, evidence: measured}
    right_hand: {width: 6, source: {feature: hand,  columns: [6, 12]},  evidence: measured}
    right_leg:  {width: 6, source: {feature: robot, columns: [6, 12]},  evidence: constant}
    waist:      {width: 3, source: {feature: robot, columns: [12, 15]}, evidence: measured}
```

순서는 규약이 정한다:

```yaml
# dataset_registry/layouts/gr1_body_parts.yaml
order: [left_arm, left_hand, left_leg, neck, right_arm, right_hand, right_leg, waist]
```

슬롯은 prefix sum으로 나온다. 검증은 세 가지:

- `blocks`의 키 집합이 `order`의 집합과 정확히 일치할 것 (빠짐도 잉여도 오류)
- 폭의 합이 선언된 `width`와 같을 것
- 소스 열 범위의 폭이 블록 폭과 같을 것

**블록을 리스트가 아니라 매핑으로 두는 이유.** 데이터셋이 순서를 말할 수 있으면 레이아웃
파일의 순서와 어긋날 수 있고, 그러면 "전환은 파일 하나 수정"이 깨진다. 매핑은 순서를 표현할
수단이 없으므로 순서의 출처가 하나로 강제된다.

따라서 **폴백은 두지 않는다.** 모든 데이터셋이 레이아웃 파일을 참조한다. `galaxea`처럼 공유
체계가 없는 경우도 자기 레이아웃 파일(`layouts/galaxea.yaml`)을 갖는다.

**얻는 것.** 34폭 `agibot_gripper`는 지금 손 블록만 1칸인 별개의 손수 타일링인데, 이 구조에서는
`left_hand: {width: 1}` 한 줄이고 34는 계산 결과다. `humanoid_everyday`의 family B도
`order: [left_arm, right_arm, left_hand, right_hand]` 파일 하나로 표현되어, 별개 계열이 아니라
**순서가 다른 같은 규약**이 된다.

## 재구축 가능성 판정

블록마다 소스가 해결되어야 값을 채울 수 있다. 레지스트리에 물어본 현황:

| 데이터셋 | state | action | 판정 |
|---|---|---|---|
| `action_net` | 44/44 소스 있음 | 44/44 | 가능 |
| `agibot_dexhand` | 30 + 14 zero | 30 + 14 | 가능 (zero 합성) |
| `agibot_gripper` | 20 + 14 zero | 20 + 14 | 가능 |
| `galaxea` | 18/18 | **26 전부 `unknown`** | action 불가 |
| `humanoid_everyday_g1` | 28/28 | 28/28 | 가능 |
| `neural_robocurate` | **0/44** | 0/44 | 불가 (upstream 없음) |

규칙:

- `constant` + 소스 없음 → **zero 합성.** 로봇에 없는 부위라 값이 정의상 0이다.
- `unknown` / `declared` / `inferred` + 소스 없음 → **로드 실패.** 조용히 0을 채우면 galaxea의
  action 26칸이 0인 데이터셋이 나온다. 그건 학습을 망가뜨리면서 성공한 것처럼 보인다.
- `plan` 커맨드가 이 판정을 실행 전에 먼저 출력한다.

`evidence` 필드는 역산 근거를 남기려고 도입했는데, 결과적으로 재구축 가능성 판정을 겸한다.

## `state_layout` — 첫 table step

### 왜 컨버터가 아니라 step인가

컨버터가 처음부터 44슬롯을 쓰면 패스가 하나로 끝나 빠르다. 그런데:

- `galaxea` / `humanoid_everyday`처럼 upstream이 이미 LeRobot인 소스는 훅을 걸 컨버터가 없다.
  같은 규칙을 재배치 전용 코드로 한 번 더 쓰게 된다.
- canonical 전환 때 영상까지 다시 굽는다. ActionNet 기준 2.49 TiB 재다운로드 + 전체 재변환 +
  전체 재인코딩.

step으로 두면 두 경로가 같은 코드를 지나고, 전환은 parquet만 다시 쓴다. ActionNet 30,120
에피소드 × 약 400행 × 44 float은 수백 MB 규모다.

### 계약

`resize`가 구현하는 `VideoStep`과 나란히 `TableStep` 프로토콜을 둔다.

```
kind = "table"
apply(root: Path, out: Path, spec) -> None
```

- 입력: LeRobot 데이터셋 디렉토리 (v2.1 / v3.0 둘 다)
- 하는 일: 에피소드 parquet에 `observation.state`, `action` 열을 쓴다.
  `slot_map("state")` / `slot_map("action")`이 유일한 입력이다.
- 부수 산출물: `meta/modality.json` 생성, `meta/info.json`의 features 갱신
- **영상 파일은 하드링크로 통과**시킨다. 프레임 수가 바뀌지 않으므로 안전하다.

### `frame`이 아니라 `table`인 이유

`lerobot_pipeline/README.md`가 예약해둔 `frame` 종류는 "프레임 수를 바꾸거나 parquet 열을
건드리는 것"으로 뭉뚱그려져 있다. 이 둘은 난이도가 다르다.

| 종류 | 프레임 수 | 영상 처리 | 상태 |
|---|---|---|---|
| `table` | 보존 | 하드링크 | 이번에 구현 |
| `frame` | 변경 | 재분할 필요 | 계속 예약 |

### 통계

재계산하지 않고 **치환**한다. 모든 슬롯이 소스 피처 열의 복사본이므로 per-column 통계
(mean / std / min / max / count)를 그대로 옮기면 정확히 일치한다. ActionNet은 constant
블록(다리)까지 전부 `robot` 소스를 갖고 있어 44슬롯 전체가 치환으로 해결된다.

소스가 없는 zero 블록(`agibot`의 다리)만 상수 통계를 합성한다: mean = min = max = 0,
std = 0, count는 프레임 수. 30,120 에피소드 전체 재순회를 피한다.

---

# 규약과 실행

## 규약 파일이 곧 "처리 방식 일람"

state 배치 규칙과 video 처리가 한 파일에 나란히 놓인다. RLDX-1 규약 전체가 이 파일이다.

```yaml
# lerobot_pipeline/configs/profiles/rldx1.yaml
state:
  layout: gr1_body_parts
video:
  resize: {type: resize_preserve_aspect_area, max_area: 65536, multiple: 32}
  encoding: rldx1_reference
dest:
  version: lerobot_v21
```

실행 config는 여기에 경로만 붙인다:

```yaml
# lerobot_pipeline/configs/actionnet_rldx1.yaml
name: actionnet_rldx1
dataset: action_net
profile: rldx1
source:
  path: /scratch/data
  args: {executor: local, episodes_per_task: 100}
dest:
  path: /scratch/out/actionnet_rldx1
```

`source.type`은 레지스트리에서 온다. `steps` / `dest.version` / `runtime.encoding`을 실행
config에 직접 써서 profile을 덮어쓸 수도 있다 — 기존 config는 profile 없이도 그대로 동작한다.

`profile.state.layout`은 데이터셋 yaml의 `state.layout`을 덮어쓴다. canonical 전환은
`rldx2.yaml`에 다른 레이아웃 이름을 적는 것으로 끝난다.

## 해석 결과 보기

3층으로 나눈 대가로 "이번 실행이 실제로 뭘 하는가"가 흩어진다. 해석된 결과를 한 화면에
펼치는 커맨드를 둔다.

```bash
python -m lerobot_pipeline.plan --config lerobot_pipeline/configs/actionnet_rldx1.yaml
```

출력에 들어가는 것: 재구축 가능성 판정, 스테이지 순서, 소스 서술(포맷·경로·시계 전략),
슬롯 맵 전체(슬롯 → 소스 열), 비디오 필터와 인코딩 파라미터, 최종 출력 버전. 실행 없이
계산만 하므로 config 검토용이다.

## 스테이지 순서

```
convert  →  state_layout  →  transform(video)  →  version_convert
   ↑            ↑                  ↑                    ↑
소스 피처만    44슬롯 조립       area 256²          v3.0 → v2.1
```

기존 LeRobot이 소스인 경우(`humanoid_everyday`)는 `convert`가 없고 `state_layout`부터
시작한다. 같은 step이 두 경로를 모두 처리한다.

`state_layout`을 `transform`보다 앞에 두는 이유는 둘이 서로 다른 파일을 만지므로 정합성
문제는 없고, 다만 실패했을 때 값싼 쪽(parquet)이 먼저 끝나 있는 편이 낫기 때문이다.
`version_convert`보다 앞인 것은 필수다 — v3.0에서 열을 정리한 뒤 내려가야 한다.

---

# 무손상 증명

새 실데이터 없이 지금 레포 안에서 네 겹으로 본다.

**(a) 타일링 골든 테스트.** 지금 yaml에 손으로 적힌 `slots` 값이 그대로 기준이다. 새 구조에서
계산한 슬롯이 6개 데이터셋 전부에 대해 한 칸도 다르지 않아야 한다. 골든 값은 리팩터링 전에
테스트 파일에 박아 넣는다.

**(b) 조립 등가 테스트.** `assemble()`을 지우기 전에, 무작위 `robot`(32) / `hand`(12)에 대해
`assemble()` 결과와 `slot_map()`으로 만든 44벡터가 같음을 확인한다. 통과한 뒤 지운다.

**(c) 리더 등가 테스트.** 합성 hdf5 에피소드(로봇 60 Hz / 영상 30 Hz, 겹치는 타임스탬프와
비단조 구간 포함)를 만들어, `actionnet_utils.load_episode`와 `spec2lerobot`의
`hdf5_episodes` + `nearest_timestamp_dedup`이 같은 행 인덱스와 같은 소스 피처를 내는지
확인한다. 통과한 뒤 `actionnet2lerobot/`을 지운다.

**(d) 소형 골든 데이터셋.** 3에피소드짜리 합성 LeRobot 데이터셋을 `state_layout`에 통과시키고
parquet 열, `modality.json`, 통계를 검사한다. 레포에 커밋 가능한 크기다.

기존 `dataset_registry/tests/test_registry.py`의 `test_slot_map_agrees_with_the_converter`는
`PERMUTATION`이 사라지므로 (b)로 대체된다.

## 인수 기준

- `action_net`을 처리하는 데 필요한 정보 중 **코드에 남은 것이 없다.** 데이터셋 이름이
  `dataset_registry/datasets/action_net.yaml` 밖의 어떤 `.py` 파일에도 등장하지 않는다.
  (테스트 픽스처는 예외)
- `plan` 커맨드가 6개 데이터셋 각각에 대해 재구축 가능 여부를 정확히 판정한다.
- 전체 테스트 통과.

---

# 파일 단위 변경

## 신규

| 파일 | 내용 |
|---|---|
| `dataset_registry/layouts/gr1_body_parts.yaml` | 블록 순서 |
| `dataset_registry/layouts/arms_then_hands.yaml` | 블록 순서 |
| `dataset_registry/layouts/galaxea.yaml` | 블록 순서 |
| `spec2lerobot/__main__.py` | CLI |
| `spec2lerobot/adapter.py` | `SpecAdapter(BaseAdapter)` |
| `spec2lerobot/formats/hdf5_episodes.py` | 포맷 리더 |
| `spec2lerobot/formats/__init__.py` | `register_format` / `build_format` |
| `spec2lerobot/clocks.py` | `register_clock`, `nearest_timestamp_dedup`, `index` |
| `spec2lerobot/tests/` | (c) 리더 등가 테스트 |
| `lerobot_pipeline/steps/state_layout.py` | table step |
| `lerobot_pipeline/profiles.py` | profile 로더 (`encoding.py`와 같은 패턴) |
| `lerobot_pipeline/plan.py` | 해석 결과 출력 CLI |
| `lerobot_pipeline/configs/profiles/rldx1.yaml` | RLDX-1 규약 |

## 변경

| 파일 | 변화 |
|---|---|
| `dataset_registry/schema.py` | 블록이 `slots` 대신 `width`, layout 로더, `source:` 섹션, 슬롯 계산·검증, 재구축 가능성 판정 |
| `dataset_registry/datasets/*.yaml` | 6개 전부 재작성 (슬롯 제거, `source:` 추가) |
| `dataset_registry/README.md` | 레이어 설명 |
| `lerobot_pipeline/registry.py` | `TableStep` 프로토콜 |
| `lerobot_pipeline/stages.py` | `state_layout` 스테이지, `spec2lerobot` 컨버터 |
| `lerobot_pipeline/config.py` | `dataset:` / `profile:` 키, profile 해석 |
| `lerobot_pipeline/README.md` | 3층 구조, `table` step |
| `lerobot_pipeline/configs/actionnet_rldx1.yaml` | `actionnet_v21.yaml` 대체 |
| `pyproject.toml` | `pytest` dev 의존성, `spec2lerobot` extras |

## 삭제

| 파일 | 이유 |
|---|---|
| `actionnet2lerobot/` 전체 (622줄) | 데이터로 대체 — (c) 통과 후 |

## 손대지 않음

`verify.py`, `encoding.py`, `transform.py`, `video_ops.py`, `meta.py`, `steps/resize.py`,
`generic_converter/`, 그리고 ActionNet 외의 모든 컨버터.

`verify.py`가 그대로 남는 것은 우연이 아니라 제약이다. 이 파일은 `spec.blocks`를 순회하며
`block.start` / `block.end` / `block.width` / `block.evidence` / `block.feature` /
`block.src_start`를 읽는다. 따라서 **`Block` 데이터클래스의 필드와 `StateSpec.blocks`가
정렬된 튜플이라는 점은 유지한다** — yaml 표현만 바뀌고 로드된 객체 모양은 그대로다.
`slot_map()`의 시그니처도 유지한다. 이 제약이 스키마 리팩터링 범위를 로딩부에 가둔다.

---

# 구현 중 설계에서 달라진 것

설계 후 실제로 만들면서 바뀐 것들. 전부 코드를 쓰다 발견한 사실 때문이다.

**`pad` 필드를 추가했다.** agibot의 2-DoF 헤드가 스켈레톤의 3칸 `neck` 블록에 들어간다.
설계에는 이 경우가 없었다. 기존 yaml은 `neck` + `neck_pad` 두 블록으로 쪼개 표현했는데,
블록 이름이 레이아웃과 정확히 일치해야 하는 새 구조에서는 불가능하다. 명시적 `pad: 1`로
표현하고, 소스 범위 + pad = 블록 폭을 검증한다. 산술이 우연히 맞아떨어지는 것과 의도한
패딩을 구분한다.

**`source_features`가 두 개의 네임스페이스였다 — 그리고 이것은 PR #2의 버그였다.**
`action_net.yaml`은 `state/robot`(hdf5 경로)을 적고 있었고, 다른 스펙들은
`observation.joint_position`(LeRobot 열 이름)을 적고 있었다. `verify.py`는 후자로
해석하므로, action_net 검증은 **모든 슬롯을 조용히 건너뛰고 OK를 냈다.** 둘을 갈랐다:
`source.features`는 어디서 읽는지, `lerobot.state.source_features`는 무엇으로 내보내는지.

**`source.feature_widths`를 추가했다.** 레이아웃에서 파생한 "블록이 읽는 최대 열"로는
소스의 진짜 폭을 알 수 없다. 0..31을 읽는다는 것이 배열이 32열인지 44열인지 말해주지
않는다. GR2의 29관절과 24값 핸드를 걸러내던 검사가 이것 없이는 약해진다.

**profile의 `state.layout`이 `state.layouts` 치환 맵이 되었다.** 단일 이름으로는 "어떤
데이터셋의 레이아웃을 무엇으로 바꾸는가"를 표현할 수 없다. 치환 맵은 rldx1에서 항등이고,
전환은 오른쪽 값을 바꾸는 것이다. 블록 이름이 다른 레이아웃으로의 치환은 거부된다 —
재정의가 아니라 재정렬이어야 한다.

**`verify.py`를 한 줄 고쳤다.** `range(block.width)` → `range(block.sourced_width)`.
`pad`가 생기면서 블록 폭과 소스 폭이 갈라졌기 때문이다. "손대지 않는다"고 적었던 것이
틀렸다. 나머지 제약(Block 필드, 정렬된 튜플, slot_map 시그니처)은 지켰다.

**등가 테스트 (c)를 살아있는 import가 아니라 고정 상수로 만들었다.** 삭제한 패키지를
import하는 테스트는 삭제 커밋 이후 영원히 skip된다 — 죽은 코드다. 옛 컨버터의 44원소
치환을 테스트 안에 리터럴로 박고, 리더 출력을 slot_map으로 조립한 것과 비교한다. 옛
코드 없이도 성립한다.

**`layouts/galaxea.yaml` 대신 `galaxea_r1.yaml`과 `unrecovered.yaml`.** galaxea의
action 26칸은 state와 다른 벡터라 별도 레이아웃이 필요했다. `unrecovered`는 "폭은 알고
구성은 모르는" 벡터를 위한 재사용 가능한 단일 블록 레이아웃이다.

**`lerobot.video.shape`를 추가했다.** 컨버터의 `RGB_SHAPE = (800, 1280, 3)`이 갈 곳이
설계에 없었다.

**pytest는 extras가 아니라 dependency group으로.** `uv run pytest`가 바로 동작한다.

# 열린 문제

- `neural_robocurate` / `march_robocurate` 7개(85,174 에피소드)는 upstream이 존재하지 않는다.
  ALIN Lab이 GR1 녹화본에서 생성한 파생 데이터고 Notion Q13에 "재현이 안될 것 같음"으로 답이
  있다. 이 설계에서는 "재구축 불가"로 정확히 판정되지만, 납품본 재가공이냐 제외냐는 결정이
  남는다.
- `galaxea`의 action 26칸은 소스가 밝혀지지 않았다. state 18칸만으로는 데이터셋을 만들 수
  없으므로 upstream 원본을 열어 열 대응을 찾아야 한다.
- 레지스트리의 공개 소스 4개(`action_net`, `agibot` ×2, `galaxea`)가 모두
  `commercial_use: false`다. 작업 자체와는 무관하나 우선순위 판단에 영향이 있다.
