"""``reader`` 와 ``images`` 가 함께 쓰는 예외.

둘이 서로를 import 하기 때문에(리더가 프레임을 디코드하고, 디코더가 못 읽은 프레임을
건너뛰기로 알린다) 공통 예외를 어느 한쪽에 두면 순환 import 가 된다. 그래서 아무것도
import 하지 않는 이 모듈에 둔다.
"""


class EpisodeSkipped(Exception):
    """One episode is unusable. Carries the reason so a run can log it."""
