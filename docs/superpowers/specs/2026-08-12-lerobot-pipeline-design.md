# LeRobot 전처리 파이프라인 설계

작성일: 2026-08-12

## 배경

RLDX(g0) 모델은 정사각형으로 리사이즈된 이미지가 아니라 **원본 aspect ratio를 유지한** 입력을 받는다.
학습 코드에서 학습 중 resize 기능은 강제로 꺼져 있으므로, **데이터셋 자체가 이미 리사이즈되어 있어야 한다.**

요구되는 전처리 로직:

- aspect ratio를 유지하며
- `resized H*W <= 256^2` 이 되도록
- `H`, `W`가 32(image patch size)의 배수가 되도록 center crop

출처: Slack 스레드 `https://rlwrld.slack.com/archives/C091PM4LKHB/p1773915494928419`
(2026-03-19 시작, 전처리 관련 내용은 2026-03-23~24 답글). 스레드에 참조 구현 두 개가 있다.

1. 목표 크기를 계산하는 순수 함수 `resize_preserve_aspect_area_then_crop(h, w, max_area, m)`
2. LeRobot mp4를 ffmpeg `scale=W:H`로 재인코딩하는 병렬 스크립트

지금은 이런 전처리를 할 때마다 일회성 스크립트를 쓴다. 목표는 **전처리 흐름을 config 한 장으로 기술하고, CLI가 그 config를 읽어 순차 처리한 뒤 최종본을 저장하는 것**이다.

## 목표

- 전처리 단계를 config로 선언하고 순서대로 실행한다.
- 이 레포가 제공하는 모든 source를 대상으로 한다 (raw 데이터셋 + 이미 LeRobot인 데이터셋).
- 최종 출력 버전을 config로 지정한다 (v2.1 / v3.0).
- 고사양 서버 한 대에서 가용 코어를 최대한 활용해 시간을 줄인다.
  (여러 대가 필요한 규모는 tar 단위 샤딩으로 푼다 — 「실행 아키텍처」)
- 새 전처리 작업을 추가할 자리를 구조적으로 남긴다.

## 비목표 (v1에서 하지 않음)

- frame 단위 스텝 엔진 (`subsample_frames` 등) — 인터페이스만 정의하고 구현하지 않는다.
- `openx` / `robocasa` / `robomind`의 `BaseAdapter` 포팅.
- v2.1 writer 직접 구현 (기존 `v30_to_v21` 스크립트를 재사용한다).
- GPU(NVENC) 인코딩 경로.
- Ray / Slurm 스케일 아웃.
- 파이프라인 코어의 S3 입출력. 코어는 디렉토리 → 디렉토리만 다루고, S3는 별도 오케스트레이터
  스크립트가 감싼다 (「실행 아키텍처」 참고).

## 설계 근거

### 왜 프레임 재작성 엔진이 아니라 mp4 직접 처리인가

v1의 유일한 스텝인 resize는 **비디오만 건드린다.** parquet의 state/action은 전혀 바뀌지 않는다.
전체 프레임을 디코딩해서 데이터셋을 통째로 재작성하면 싼 일을 비싼 방법으로 하게 된다.

| 방식 | 데이터 통과 | 중간본 디스크 |
|---|---|---|
| v21→v30 → 프레임 재작성 → v30→v21 | 3패스 | 원본 2벌 |
| mp4 직접 처리 | 1패스 | 0벌 |

확장성은 프레임 재작성 엔진이 아니라 **config 스키마 · stage 분리 · 스텝 레지스트리라는 구조**에서 나온다.
구조는 갖추되 엔진은 실제로 frame 스텝이 필요해질 때 추가한다.

### 왜 video 스텝만이면 버전 변환을 생략하는가

v2.1이든 v3.0이든 비디오는 mp4이고 리사이즈 방법이 동일하다.
`lerobot_v21 → lerobot_v21` 에 video 스텝만 있으면 v3.0을 경유할 이유가 없다.
경유를 피하면 부수 이득도 있다 — `v30_to_v21`의 `-c:v copy` 세그먼트 분할은 키프레임 경계에서
프레임이 어긋날 수 있는 기존 동작인데, 경유하지 않으면 이 리스크를 지지 않는다.

frame 스텝이 섞이는 순간에만 v3.0 허브를 경유한다.

### 화질

검증 결과 버전 변환은 재인코딩하지 않는다.

- `ds_version_convert/v30_to_v21/convert_dataset_v30_to_v21.py:335` — `-c:v copy` (stream copy)
- `ds_version_convert/v21_to_v30/convert_dataset_v21_to_v30.py` — `concatenate_video_files` (stream copy)

실제 재인코딩은 transform 1회뿐이므로 어느 경로를 택하든 손실 세대는 1세대로 동일하다.

## 아키텍처

```
config 1장
   │
   ├── ① source stage    ─→  LeRobot 데이터셋
   │      lerobot_v21 / lerobot_v30              : 통과 (변환 없음)
   │      agibot / libero / openx / robocasa
   │      / robomind                             : 기존 컨버터 CLI를 subprocess 호출 → v3.0
   │
   ├── ② steps stage     ─→  LeRobot 데이터셋
   │      config의 steps를 순서대로 적용
   │
   └── ③ dest stage      ─→  최종본
          버전이 이미 맞으면 통과
          아니면 기존 ds_version_convert 스크립트 호출
```

모든 stage는 **디렉토리 → 디렉토리** 계약을 갖는다. `ds_version_convert/*`가 이미 이 형태라 그대로 감쌀 수 있다.

컨버터를 subprocess로 호출하는 이유: `openx`는 `tfds`, `robocasa`는 `robosuite` 등 무겁고 서로 충돌할 수 있는
선택적 의존성이 있다. 프로세스를 분리하면 안전하고, 컨버터별로 다른 env를 쓸 여지도 남는다.

### stage 순서 최적화

**video 스텝은 버전 변환보다 먼저 실행한다.** 결과는 동일하고 비용이 낮다.

- 리사이즈 후에 버전 변환하면 변환이 다루는 데이터가 작아진다.
- v2.1 소스는 에피소드당 mp4가 하나라 파일 수가 많고, 변환(concat) 전에 처리하면 병렬도도 높다.

### 중간 산출물

작업 디렉토리에 쌓이고 최종본만 남긴다. `--keep-intermediate`로 디버깅 시 보존한다.

## Config 스키마

```yaml
name: humanoid_everyday_g1_rldx      # 로그·작업 디렉토리 명명용

source:
  type: lerobot_v21                  # lerobot_v21 | lerobot_v30 | agibot | libero
                                     # | openx | robocasa | robomind
  path: ~/data/vla_pretrain_dataset/humanoid_everyday/humanoid_everyday_g1
  args: {}                           # 컨버터 CLI 추가 인자 (type이 컨버터일 때만)

steps:
  - type: resize_preserve_aspect_area
    max_area: 65536                  # 256 ** 2
    multiple: 32                     # image patch size
    keys: null                       # null이면 모든 비디오 키

dest:
  type: lerobot_v21                  # lerobot_v21 | lerobot_v30
  path: ~/data/vla_pretrain_dataset/humanoid_everyday/humanoid_everyday_g1_rldx

runtime:                             # 전부 선택. 생략 시 자동 결정
  workers: auto
  threads_per_ffmpeg: auto
  preset: null                       # null이면 소스 인코딩 설정을 따름
  crf: null
```

컨버터가 source인 경우:

```yaml
source:
  type: openx
  path: ~/data/openx/bridge
  args: { fps: 5, robot_type: widowx }
```

실행:

```bash
python -m lerobot_pipeline.run --config configs/humanoid_everyday_g1_rldx.yaml
```

## 스텝

### 코드 지정 방식

**레지스트리 이름만** 사용한다. 임의 import 경로는 받지 않는다.
config 검증에서 오타를 즉시 잡을 수 있고, 사용 가능한 스텝 목록이 등록부 한 곳에 모인다.

### 스텝 종류

| 종류 | 계약 | v1 |
|---|---|---|
| `video` | mp4 직접 처리 + `info.json` shape 갱신 | 구현 |
| `frame` | 전체 디코딩 후 데이터셋 재작성 | 인터페이스만 |

### video 스텝 인터페이스

스텝은 **무엇을 할지 계획만 반환**하고 실행은 runner가 한다.

```python
class VideoStep(Protocol):
    kind = "video"

    def plan(self, shape: tuple[int, int]) -> VideoPlan | None:
        """shape: 해당 비디오 키의 현재 (H, W).
        반환: ffmpeg 필터 문자열 + 결과 (H, W). None이면 이 키는 건너뜀."""
```

계획과 실행을 분리하는 이유:

- **여러 video 스텝이 있어도 필터가 하나로 합쳐져 여전히 1패스**다 (`scale=...,crop=...`).
- 최종 shape을 실행 전에 확정할 수 있어 `info.json` 패치를 미리 계산한다.
- no-op 감지(목표 shape == 현재 shape)가 실행 없이 가능하다.

### resize_preserve_aspect_area

Slack 참조 함수를 그대로 사용한다.

```python
(h_r, w_r), (h_c, w_c) = resize_preserve_aspect_area_then_crop(h, w, max_area, multiple)
# ffmpeg filter: f"scale={w_r}:{h_r},crop={w_c}:{h_c}"   (crop 기본값이 center)
```

**알려진 경계 조건**: `short_r = max(m, ...)` 때문에 짧은 변이 이미 `multiple`(32px) 이하이면
면적 상한이 깨진다 (예: 32×4096 → 면적 131072 > 65536).
로봇 카메라에서 나올 수 없는 해상도이고 알고리즘은 모델팀이 실제 학습에 쓴 것과 동일하게 유지해야 하므로,
**알고리즘은 그대로 두고 결과 면적을 검증해 상한 초과 시 에러**를 낸다.

## 성능

H.264는 트랜스폼 도메인 다운스케일이 없으므로 **전체 디코드+인코드가 물리적 하한**이다.
목표는 그 하한에 코어 수만큼 붙는 것.

### 스레드 자동 조절

v2.1은 에피소드당 mp4가 하나라 파일이 수천~수만 개지만, v3.0은 비디오를 큰 청크로 이어붙여 파일 수가
적다. 파일 단위 병렬만으로는 v3.0에서 코어를 다 못 쓴다.

→ 파일 수와 코어 수를 보고 자동 결정한다.

```
threads_per_ffmpeg = clamp(cores / files, 1, 8)
workers            = cores / threads_per_ffmpeg
```

v2.1(파일 다수)은 1스레드 × N프로세스로, v3.0(파일 소수)은 다스레드 × 소수 프로세스로 갈린다.

### LPT 스케줄링

에피소드 길이가 제각각이라 파일 크기 편차가 크다. 무작위 순서로 던지면 마지막에 큰 파일 하나를 혼자 기다린다.
**크기 내림차순으로 정렬해 투입**한다. 비용이 거의 없고 꼬리 시간을 크게 줄인다.

### 하드링크

손대지 않는 것은 전부 하드링크로 연결한다 — parquet, meta, **그리고 리사이즈 대상이 아닌 비디오 키까지.**
같은 파일시스템이 아니면 복사로 폴백한다. 원본은 어떤 경우에도 in-place로 수정하지 않는다.

### no-op 감지와 재개

- 목표 shape이 현재 shape과 같으면 재인코딩하지 않고 하드링크한다.
- 중단 후 재실행 시, 이미 있는 출력 파일은 `ffprobe`로 해상도·프레임 수를 검증한 뒤 건너뛴다.

### GOP (중요)

**재인코딩 시 GOP(키프레임 간격)를 소스와 동일하게 유지해야 한다.**

LeRobot은 학습 중 비디오에서 랜덤 프레임을 디코딩한다. GOP가 길면 프레임 하나를 얻으려고
직전 키프레임부터 디코딩해야 해서 **데이터로더가 느려진다.**
ffmpeg 기본 GOP는 250프레임이라, 지정하지 않고 재인코딩하면 변환은 성공했는데
**학습 단계에서 뒤늦게 성능 저하로 나타난다.**

→ source `info.json`의 비디오 인코딩 파라미터를 읽어 동일하게 맞춰 재인코딩하고, 읽을 수 없으면 경고를 낸다.
Slack 스크립트는 이 값을 지정하지 않았다.

### 기타

- `-an` (오디오 없음). 로봇 비디오에 오디오 트랙이 없다.
- `preset` / `crf`는 소스 설정을 따르는 것이 기본. 벤치마크로 측정해 조정할 수 있게 열어둔다.

### 벤치마크 서브커맨드

"최대한"의 정답이 소스 해상도·스토리지 대역폭에 달려 있어 측정이 필요하다.

```bash
python -m lerobot_pipeline.bench --config configs/xxx.yaml --sample 20
# → preset/threads 조합별 실측 처리량 + 전체 예상 시간
```

### 실측 (2026-08-12)

`s3://rlwrld-foundry-data/external/action_net/.../01JGJCNDDF-01JGJY03Q1.tar` (4.41GB, `rgb.mp4` 100개)
중 24개(10,597 프레임, `800x1280` → `192x288`)로 측정. 두 머신 모두 물리 8코어, ffmpeg 6.1.1-3ubuntu5 동일 빌드.

| 머신 | workers × threads | wall | fps | fps/물리코어 |
|---|---|---|---|---|
| c7gd.2xlarge (Neoverse-V1, 8 vCPU) | 8 × 1 | 21.3s | 498 | 62.3 |
| | 8 × 8 | 19.8s | 535 | 66.9 |
| | 1 × 8 | 26.6s | 399 | 49.8 |
| c6id.4xlarge (Xeon 8375C, 16 vCPU) | 8 × 1 | 19.1s | 556 | 69.5 |
| | 16 × 1 | 12.3s | 862 | 107.7 |
| | 16 × 16 | 11.5s | 918 | 114.8 |
| | 1 × 8 | 20.3s | 522 | 65.2 |

무엇이 참으로 확인되고 무엇이 틀렸는지:

* **파일 단위 병렬 > 파일 내부 스레딩.** `1 × 8`이 양쪽 모두 최하위다. 워커를 늘리는 설계 방향은 옳다.
* **스레드 과다구독은 이 워크로드에서 문제가 아니었다.** 설계 초안은 이것을 과거 작업이 느렸던
  주원인으로 지목했으나, 측정 결과 과다구독(8×8, 16×16)이 오히려 약간 빨랐다.
  출력이 192×288로 작아 x264가 지정한 스레드를 실제로 다 쓰지 못하고, 남는 스레드는 경합 없이 유휴로
  남기 때문이다. `-threads` 고정은 예측 가능성 때문에 유지할 가치가 있지만 성능 개선 수단은 아니다.
* **아키텍처 간 코어당 격차는 12%** (556 vs 498). 초안이 추측한 20~40%보다 훨씬 작다.
* **x86의 우위는 SMT에서 나온다.** 8×1 → 16×1 에서 +55%.
* **auto 플래너는 프로덕션에서 near-best를 고른다.** `os.cpu_count()`(=16)로 계획하면 16×1(862 fps)을
  선택해 최적값의 6% 이내다.

비용 (us-east-1 온디맨드 대략):

| | 최고 fps | 시간당 | 100만 프레임당 비용 | 100만 프레임당 시간 |
|---|---|---|---|---|
| c7gd.2xlarge | 535 | ~$0.363 | $0.19 | 31분 |
| c6id.4xlarge | 918 | ~$0.806 | $0.24 | 18분 |

속도는 x86이 1.7배, 비용은 ARM이 23% 저렴하다. 반복 실험 속도가 중요하므로 c6id 계열을 권하되,
데이터가 수십 TB로 커져 비용이 지배적이 되면 ARM으로 전환할 근거가 충분하다.

전송 비중:

| | 다운로드 (4.41GB) | 압축 해제 | 리사이즈 (100파일 추정) | 전송 비중 |
|---|---|---|---|---|
| c7gd.2xlarge | 24s | 13s | ~83s | 31% |
| c6id.4xlarge | 14s | 4s | ~48s | 27% |

출력은 719MB → 77MB (9.3배 축소). 업로드는 무시할 수준이고 다운로드가 전부다.

이 27~31%는 **순차 실행일 때의 손실이다.** 다운로드와 인코딩을 겹쳐 돌리면 첫 tar를 제외한
전송이 전부 숨는다 — 「실행 아키텍처」의 오버랩 구조가 이 표를 근거로 한다.

### 인스턴스 크기 선택

파일 단위로 병렬화하므로 워커 수가 파일 수를 넘으면 남는 코어는 유휴가 된다. 파일 크기 편차로 인한
straggler를 상쇄하려면 **워커 1개당 파일 4~8개**가 필요하다. `rgb.mp4` 100개 규모면 16~32 워커
(`c6id.4xlarge` ~ `c6id.8xlarge`)가 적정하다.

c6id는 128 vCPU(c6id.32xlarge)까지 올라가지만, tar 하나(파일 100개)에서는 코어가 병목이 아니라
파일 수와 전송 시간이 상한을 만든다. 128 vCPU가 값을 하는 것은 tar 여러 개를 한 박스에 몰아
파일이 수천 개가 될 때다 — 실제 배치가 정확히 그 조건이므로(「실행 아키텍처」), 큰 인스턴스가 정당화된다.
이 스케일링 곡선은 아직 측정하지 않았다 — 더 많은 파일로 측정할 때
`lerobot_pipeline/scripts/bench_raw_videos.py`를 그대로 쓸 수 있다.

## 메타데이터 갱신

- `features[<video_key>]["shape"]` → 새 `(H, W, C)`
- 비디오 키의 `info` 블록(`video.height` / `video.width` 등) → 새 값
- **image stats(mean/std)는 기본적으로 갱신하지 않는다.** 리사이즈로 미세하게 달라지지만
  정확한 재계산은 전체 디코딩이 필요하다. `--recompute-stats`로 열어두되 기본 off.

## 에러 처리

- ffmpeg 실패 파일을 모아 보고하고, **하나라도 실패하면 stage 실패 + dest 삭제.**
  반쯤 처리된 데이터셋이 성공한 것처럼 남으면 안 된다.
- 등록되지 않은 step type → 사용 가능한 목록과 함께 즉시 에러.
- `dest.path`가 이미 존재하면 기본 거부, `--overwrite` 필요.
- 면적 상한 위반(위 경계 조건) → 에러.

## 파일 배치

```
lerobot_pipeline/
├── README.md
├── __init__.py
├── config.py        config 스키마 · YAML 로더 · 검증
├── registry.py      스텝 레지스트리 (이름 → 구현)
├── steps/
│   ├── __init__.py
│   └── resize.py    resize_preserve_aspect_area
├── video_ops.py     ffmpeg 호출 · mp4 병렬 처리 · 스레드 자동 조절
├── meta.py          info.json 패치 (v2.1 / v3.0 공통)
├── stages.py        stage 조립 및 경로 배선
├── run.py           CLI 엔트리포인트 (디렉토리 → 디렉토리)
├── bench.py         벤치마크 서브커맨드
├── scripts/
│   ├── s3_batch.py           S3 오케스트레이터 (다운로드·처리·업로드 오버랩, 샤딩)
│   └── bench_raw_videos.py   원본 mp4 처리량 측정
└── configs/
    └── humanoid_everyday_g1_rldx.yaml
```

## 테스트

- **순수 함수**: `resize_preserve_aspect_area_then_crop` — Slack 예시 `(1280,720) → ((192,256),(192,256))`,
  면적 상한, 32의 배수, 업스케일 없음, 경계 조건 에러.
- **end-to-end**: 작은 합성 v2.1 / v3.0 데이터셋 픽스처로
  mp4 실해상도, `info.json` shape, **프레임 수 불변**, **parquet 무변경**을 검증.
- **config 검증 실패**: 미등록 스텝, `dest.path` 충돌, 알 수 없는 source type.
- **재개**: 중간에 중단 후 재실행 시 완료 파일을 건너뛰고 결과가 동일한지.

## 구현 중 확인할 항목

로컬에 `lerobot`이 설치되어 있지 않아 코드로 확인하지 못한 것들이다. 구현 시 설치된 패키지로 확정한다.

- `info.json`의 비디오 관련 필드 정확한 이름 (v2.1 / v3.0 각각)
- LeRobot 기본 비디오 인코딩 파라미터 (GOP, codec, pix_fmt)
- v2.1 / v3.0의 비디오 경로 템플릿

## 실행 아키텍처 (S3 → EC2 → S3)

### 규모

처리 대상은 보통 10TB 내외, 최대 48TB다 (실측한 AgiBotWorld-Beta 46.13TB가 이 상한에 해당한다).
실측 tar 하나(4.41GB / 약 44,000 프레임 → 프레임당 약 100KB)에서 외삽하면:

| 총량 | 프레임(추정) | c6id.4xlarge (918 fps) | c6id.32xlarge (추정 ~6,900 fps) | 32xlarge ×4 |
|---|---|---|---|---|
| 10TB | 약 1억 | 30시간 | 4시간 | 1시간 |
| 48TB | 약 4.8억 | 145시간 | 19시간 | 5시간 |

컴퓨트 비용은 100만 프레임당 $0.24로 48TB 전체가 **약 $115**다. 코어 수에 선형이므로 큰 박스 1대와
작은 박스 N대의 비용이 같다. **결정 변수는 비용이 아니라 wall clock뿐이다.**

### 다운로드와 처리를 분리하지 않는 이유

로컬 NVMe는 인스턴스 전용이라 다른 인스턴스와 공유할 수 없다. 다운로드 전용 박스를 두면
네트워크 복사가 한 번 더 늘거나(A→B), NVMe를 포기하고 EBS/FSx로 내려가야 한다.

그리고 **네트워크는 병목이 아니다.** c6id.4xlarge가 918 fps를 낼 때 소비하는 입력은 약 92 MB/s
(≈0.73 Gbps)인데 실측 다운로드는 315 MB/s였다 — 처리보다 3배 빠르다. 128 vCPU로 올려도 소비량은
약 690 MB/s(5.5 Gbps)로 링크 대역 안에 여유롭게 들어간다. 전용 다운로더는 이미 남는 자원을
더 사는 것이라 아무것도 얻지 못한다.

공유 스테이징(FSx/EBS에 원본을 받아두고 재사용)도 배제한다. **원본당 전처리는 사실상 1회**라
재사용 이득이 없다.

수십 TB에서 필요한 분리는 역할 분리가 아니라 **데이터 샤딩** — 똑같은 일을 하는 박스 N개에
작업 리스트를 나눈다. 작업끼리 독립이라 처리 중 샤드 간 통신이 없다 (병합은 마지막에 한 번, 아래).

### 작업 단위

소스 타입에 따라 다르다.

| 소스 | 작업 단위 | 근거 |
|---|---|---|
| `lerobot_v21` / `lerobot_v30` | tar (또는 데이터셋 디렉토리) | 이미 LeRobot이라 그대로 처리 |
| `agibot` | **task_id** | 컨버터가 `task_info/*.json` 단위로 돈다 |
| 그 외 컨버터 | 컨버터의 자연 단위 | 어댑터의 `load_tasks()`가 정의 |

`agibot_h5.py`는 이미 `--task-ids task_327 task_351 ...`을 받는다. 샤딩은 이 리스트를 나눠 주는 것으로 끝이고
컨버터 수정이 필요 없다.

### AgiBot 소스 구조 (실측, 2026-08-12)

`agibot-world/AgiBotWorld-Beta` (HF API 전수 조사):

```
task_info/task_<ID>.json              217개
observations/<task_id>/<ep범위>.tar   1,139 tars, 46.13 TB   ← task별로 분리됨
proprio_stats/<ep범위>.tar            7 tars, 0.27 TB        ← task별이 아님
parameters/
```

두 가지가 샤딩 설계를 결정한다.

**`observations`는 task별 폴더다.** task_327만 변환하려면 `observations/327/`만 받으면 된다.
46TB 중 필요한 것만 내려받을 수 있다.

**`proprio_stats`는 task를 가로지른다.** 전역 episode ID 범위(`648533-713949.tar` 등)로 잘려 있어
한 tar에 여러 task가 섞여 있다. 다만 **전체가 7개 tar, 270GB뿐이므로 샤드 박스마다 통째로 한 번 받아
풀어둔다** (박스당 약 14분). 인터벌 매칭 로직을 짜는 것보다 단순하고, 46TB 작업 대비 무시할 수준이다.

### 샤드 분배는 LPT로

task 크기 편차가 크다.

| | 크기 |
|---|---|
| 최소 | 2.4 GB |
| 중앙값 | 152.3 GB |
| 최대 (task_362) | 1.61 TB |

670배 차이라 라운드로빈으로 나누면 샤드 하나가 몇 배를 뒤집어쓴다. **task 크기 내림차순 LPT 분배**를 쓴다
(파일 수준 LPT와 같은 원리를 샤드 수준에 한 번 더 적용). 크기는 HF/S3 API로 미리 조회할 수 있어 비용이 없다.

### 샤드 출력과 병합

**샤드 출력을 같은 S3 prefix에 올리면 서로 덮어쓴다.** 각 샤드가 만드는 것은 "전체의 일부"가 아니라
0번부터 다시 번호를 매긴 완결된 데이터셋이기 때문이다. 충돌하는 것:

| 충돌 | 이유 |
|---|---|
| `data/chunk-000/file-000.parquet` | 모든 샤드가 chunk-000/file-000부터 시작 |
| `videos/<key>/chunk-000/file-000.mp4` | 위와 동일 |
| `episode_index` · `index` · `task_index` | 샤드마다 0부터 재시작 |
| `meta/info.json` | `total_episodes` / `total_frames`가 자기 샤드 값 |
| `meta/tasks.parquet` | 같은 자연어 task가 샤드마다 다른 인덱스 |
| `meta/stats.json` | 가중 병합 필요 |
| 에피소드별 `from/to_timestamp` | 여러 에피소드가 mp4 하나에 이어붙으므로 파일 내 오프셋 재계산 필요 |

마지막 항목이 특히 파일 배치만으로는 해결되지 않는 이유다 — `generic_converter/pipeline.py`의
`src_to_offset`이 그 재계산이다.

따라서 흐름은:

```
샤드 0..N-1 → s3://.../shards/<i>/   각각 독립 prefix에 완결된 데이터셋으로            [병렬]
                      ↓
      한 박스가 N개를 받아 aggregate_datasets(roots=[...]) 1회                        [직렬]
                      ↓
              s3://.../final/                                                          [업로드]
```

**최종 산출물은 단일 LeRobot 데이터셋이어야 하므로 이 병합 단계는 생략할 수 없다.**

### 병합 박스

병합은 재인코딩하지 않는다 — `shutil.copy`와 stream copy concat뿐이라 **CPU가 아니라 IO 바운드**다.
따라서 코어 수가 아니라 **스크래치 용량과 IO 대역**으로 인스턴스를 고른다.

입력 N개와 출력 1벌을 동시에 들고 있어야 하므로 **출력 크기의 약 2배**가 필요하다.
46TB 소스 기준 출력이 약 10TB라면 스크래치 약 20TB — `c6id.32xlarge`의 NVMe 7.6TB로는 부족하다.
대용량 EBS를 붙이거나 `i4i` / `im4gn` 계열(NVMe 30TB)을 쓴다.

보통 규모(10TB 소스 → 출력 약 2.5TB)는 스크래치 5TB면 되므로 처리 박스 NVMe에 그대로 들어간다.
**병합 전용 박스는 최대 규모에서만 필요하다.**

wall clock에서 병합은 20TB 규모의 읽기·쓰기와 S3 왕복이라 46TB 기준 **약 5~7시간**으로,
변환(4대 샤딩 약 5시간)과 맞먹는다. 최대 규모에서는 여기가 병목이다.

**미측정**: 출력 크기 추정치(소스의 약 1/4~1/5)는 AgiBot 소스 해상도 640×480과 목표 면적 상한에서
계산한 픽셀 비율이다. 실제 mp4 크기는 픽셀 수에 선형이 아니고 depth 제외 여부에도 좌우되므로,
첫 실행에서 task 하나를 끝까지 돌려 실측한 뒤 병합 박스 스펙을 확정한다.

### 오버랩

전송이 wall의 27~31%인데, 순차로 하면 그 동안 코어 전체가 논다. 다운로드는 네트워크 바운드,
인코딩은 CPU 바운드라 자원이 겹치지 않으므로 **겹쳐 돌리면 첫 tar 시간만 노출되고 나머지 전송은 전부 숨는다.**

```
[다운로더] → 로컬 작업 큐(깊이 2~3) → [lerobot_pipeline.run] → [업로더] → S3 → 로컬 삭제
                    ↑ NVMe 상주량은 항상 작업 2~3개분
```

48TB를 통째로 받아둘 NVMe는 존재하지 않으므로, 이 스트리밍 구조는 최적화가 아니라 **필수 조건**이다.
AgiBot의 최대 task(1.61TB) 하나도 NVMe에 한 번에 올리기 어려우므로, 작업이 큰 소스에서는
큐 깊이를 1로 낮추거나 `--episodes-per-task`로 작업을 더 잘게 쪼갠다.

### 오케스트레이터

**`lerobot_pipeline`은 디렉토리 → 디렉토리 순수 계약을 유지한다** (비목표의 "S3 입출력은 호출자가 처리한다"
그대로). S3와 큐잉은 바깥의 별도 스크립트가 담당한다.

```
lerobot_pipeline/scripts/s3_batch.py
```

```bash
# 샤드 i (N대에서 동시에)
python -m lerobot_pipeline.scripts.s3_batch \
    --source s3://bucket/prefix/ \
    --dest   s3://bucket/out/shards/ \
    --config configs/xxx.yaml \
    --shard 0/4                    # 생략 시 단일 인스턴스, 샤드 디렉토리 없이 바로 출력

# 전부 끝난 뒤, 병합 박스에서 1회
python -m lerobot_pipeline.scripts.s3_batch merge \
    --shards s3://bucket/out/shards/ \
    --dest   s3://bucket/out/final/
```

계약과 동작:

- **작업 단위는 소스 타입이 정한다** (위 「작업 단위」 표). 나열 → 크기 내림차순 정렬(LPT) → 샤드 분할.
- **3개 스테이지가 동시에 돈다** — 다운로더, 처리(`lerobot_pipeline.run`을 subprocess로), 업로더.
  큐 깊이로 NVMe 사용량을 제한한다.
- **샤드는 각자 독립 prefix(`shards/<i>/`)에 완결된 데이터셋을 쓴다.** 같은 prefix에 쓰면 덮어쓴다.
- **`merge` 서브커맨드**가 샤드 출력을 받아 `aggregate_datasets(roots=[...])`를 1회 호출한다.
  `generic_converter.pipeline.aggregate_tasks`가 이미 `roots` 리스트만 받는 얇은 래퍼라 그대로 쓸 수 있다.
  단일 인스턴스로 돌렸으면 이 단계가 필요 없다.
- **재개**: 출력 S3에 결과가 이미 있으면 건너뛴다. 데이터셋 내부 재개는 기존 ffprobe 검증 로직을 쓴다.
- **실패 격리**: 작업 하나가 실패해도 배치 전체를 죽이지 않고 실패 목록을 모아 마지막에 보고한다.
  (데이터셋 *내부*의 "하나라도 실패하면 stage 실패 + dest 삭제" 규칙은 그대로 유지된다 — 층위가 다르다.)
  **병합은 모든 샤드가 성공해야 시작한다** — 일부만 병합하면 조용히 불완전한 데이터셋이 나온다.
- 전송은 `s5cmd`.

### 인스턴스 선택

- 보통 규모(10TB 내외): `c6id.16xlarge`~`32xlarge` **1대**, 샤딩 없이 반나절. 병합 단계가 아예 없다.
- 최대 규모(48TB): 같은 인스턴스 **4대에 샤딩**(약 5시간) **+ 병합 박스 1대**(약 5~7시간).
  **총 wall은 약 10~12시간이고 절반이 병합이다.** 샤드를 늘려도 이 절반은 줄지 않는다.
- 32xlarge 1대보다 8~16xlarge 여러 대가 나은 경우 — 스팟 가용성이 좋고 한 대가 죽었을 때 손실 범위가 작다.
  비용이 선형이라 잃는 것이 없다.
- EBS 전용 인스턴스는 피한다. 로컬 NVMe가 있는 `c6id` / `c7gd` 계열을 쓴다.

**미측정**: 위 표의 32xlarge 수치는 8/16 vCPU 실측에서 코어당 선형 외삽한 추정치다. 실제로는
파일 수가 워커 수를 못 채우거나 메모리 대역에서 꺾일 수 있다. 첫 대규모 실행 전에
`lerobot_pipeline/scripts/bench_raw_videos.py`로 tar 여러 개를 몰아 측정해 확인한다.

## 향후 확장

- `frame` 스텝 엔진 — `subsample_frames`가 실제로 필요해질 때. `LeRobotSourceAdapter(BaseAdapter)`를
  만들어 `generic_converter`의 datatrove 병렬 실행·집계를 재사용하면 된다.
  이 경우 v3.0 허브를 경유하며, runner가 스텝 종류를 보고 경로를 전환한다.
- **인덱스 오프셋 사전 할당으로 병합 제거** — 최대 규모에서 병합이 wall의 절반을 먹는데, 원인은
  샤드가 저마다 0번부터 번호를 매겨 파일명과 인덱스가 겹치는 것뿐이다. 샤드에 `episode_index` /
  `chunk_index` 범위를 미리 배정하면 비디오·parquet을 옮길 필요 없이 **메타데이터만 이어붙이면 된다.**
  병합 IO가 10TB급에서 수십 GB급으로 떨어진다.
  전제 — 샤드별 에피소드 수를 사전에 알아야 하는데, 손상 mp4나 `action_len > state_len`으로
  스킵되는 에피소드가 있어 정확히는 모른다. 실제로는 `data/*.parquet`의 `index`/`episode_index`
  컬럼만 사후 재작성하는 절충이 필요하다. 46TB 배치를 반복해서 돌리게 될 때 착수한다.
- 데이터셋 *내부* 샤딩 — 위 `s3_batch`의 `--shard`는 소스별 작업 단위다. 작업 하나(예: AgiBot의
  1.61TB task_362)가 한 박스에 안 들어가면 그 아래 수준의 분할이 필요하다. `--episodes-per-task`로
  일부 대응되지만 샤드 경계를 넘는 분할은 아직 없다.
- GPU(NVENC) 경로 — 출력 해상도가 훨씬 커지거나 CPU가 병목으로 확인될 때. 화질 특성이 x264와 달라
  모델팀 확인이 필요하다.
