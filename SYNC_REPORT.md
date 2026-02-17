# Arch ↔ Code Sync Report (v5)

이 폴더는 `architecture.md` 와 코드(Phase1~Phase4.2, Loss, Training)를 **라인 단위로 매핑**하고,
불일치 사항을 반영하여 동기화한 결과물입니다.

## 반영된 핵심 수정

- Phase1: bivector candidate(v1∧v2) 생성 및 output dict에 포함.
- Phase2: Rotor 입력 4ch→5ch(v1/v2 + bivector), scalar mixing(s_mixer) 반영.
- Phase3:
  - HIGH_RES_SKIP_LEVEL(v5) 기본값 2로 상향(옵션).
  - Skip connection(affine warp + gate + refine) 실제 forward에 반영.
  - 결과 dict에 rotor_map alias + gates(g_s,g_v,g_b) 포함.
  - 반환 results를 level 오름차순(0→coarse)으로 정렬.
- Phase3.5:
  - Feature selection rule을 architecture 기준(pos&angle 큰 경우 'B')으로 수정.
  - coarse init 선택을 results[0] 가정 대신 max(level)로 변경.
- Phase4.2:
  - priority_map 미제공 시 rotor variance + vector magnitude 기반 자동 생성 옵션 추가.
- Losses:
  - L_b를 rotor_mag 뿐 아니라 unit_cos/unit_sin orientation까지 포함하도록 확장(architecture 식과 일치).
- Training:
  - Phase3 결과에서 pred_W(A→B)를 안정적으로 추출하는 helper 추가.
  - fast_finetune도 동일 helper를 사용하도록 수정.

## 주의

- 본 폴더의 각 코드 파일 상단에는 `architecture.md` 라인 번호 기반의 매핑 주석이 포함되어 있습니다.
  (주석만 읽어도 구조를 재구현할 수 있도록 의도)
