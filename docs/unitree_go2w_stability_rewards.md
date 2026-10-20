# RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 안정성 보상함수

이 문서는 `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0` 태스크에서 **로봇 안정성(stability)** 과 직접 관련된 보상 항목을 정리합니다.

- **환경 설정**: `UnitreeGo2WRoughEnvCfg` (`rough_env_cfg.py`)
- **보상 구현**: `robot_lab/.../velocity/mdp/rewards.py` (Robot Lab 커스텀) + `isaaclab.envs.mdp.rewards` (Isaac Lab 기본)
- **가중치 0인 항목**은 `disable_zero_weight_rewards()`에 의해 학습 시 비활성화됩니다.

---

## 1. 총 보상 구조

각 스텝의 총 보상은 활성화된 보상 항(term)의 가중 합으로 계산됩니다.

$$
r_t = \sum_{i \in \mathcal{A}} w_i \cdot r_i^{\text{raw}}
$$

- $w_i$: 항목별 가중치 (weight)
- $r_i^{\text{raw}}$: 보상 함수가 반환하는 원시 값 (패널티 항은 양수 값을 반환하고, 음수 가중치가 곱해짐)
- $\mathcal{A}$: 가중치가 0이 아닌 활성 항목 집합

### 중력 기반 스케일링 (Robot Lab 공통)

다수의 보상 항은 로봇이 기울어졌을 때 기여도를 줄이기 위해 **중력 정렬 계수** $g_s$를 곱합니다.

$$
g_s = \frac{\text{clamp}(-g_{b,z},\ 0,\ 0.7)}{0.7}
$$

- $g_b = [g_{b,x},\ g_{b,y},\ g_{b,z}]^\top$: 베이스 좌표계에서의 투영 중력 벡터 (`projected_gravity_b`)
- 완전히 똑바로 선 상태: $g_{b,z} \approx -1 \Rightarrow g_s \approx 1$
- 넘어지는 중: $g_s \to 0$ (해당 항의 기여도 감소)

> `upward` 항은 $g_s$를 적용하지 않습니다.

---

## 2. 베이스(Root) 자세·운동 안정성

로봇 몸체의 수직 속도, 롤/피치 각속도, 자세 유지를 다루는 항목입니다.

### 2.1 `lin_vel_z_l2` — 수직 선속도 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-2.0** |
| 대상 | 베이스 링크 |
| 구현 | `mdp/rewards.py` |

$$
r_{\text{lin\_vel\_z}}^{\text{raw}} = g_s \cdot v_{b,z}^2
$$

- $v_b = [v_{b,x},\ v_{b,y},\ v_{b,z}]^\top$: 베이스 좌표계 선속도 (`root_lin_vel_b`)
- **목적**: 상하 바운스·도약 억제 → 지면 접촉 안정성 향상

---

### 2.2 `ang_vel_xy_l2` — 롤/피치 각속도 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-0.05** |
| 대상 | 베이스 링크 |
| 구현 | `mdp/rewards.py` |

$$
r_{\text{ang\_vel\_xy}}^{\text{raw}} = g_s \cdot \left(\omega_{b,x}^2 + \omega_{b,y}^2\right)
$$

- $\omega_b = [\omega_{b,x},\ \omega_{b,y},\ \omega_{b,z}]^\top$: 베이스 좌표계 각속도 (`root_ang_vel_b`)
- **목적**: 몸체 기울어짐(롤/피치) 진동 억제

---

### 2.3 `upward` — 똑바로 선 자세 보상

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **+1.0** |
| 대상 | 베이스 링크 |
| 구현 | `mdp/rewards.py` |

$$
r_{\text{upward}}^{\text{raw}} = \left(1 - g_{b,z}\right)^2
$$

- $g_{b,z}$: 투영 중력의 z 성분 (똑바로 서 있으면 $\approx -1$)
- **목적**: 중력 방향과 베이스 z축 정렬 유도 (유일한 **양의** 안정성 보상)
- $g_s$ 미적용

---

### 2.4 비활성 항목 (가중치 0)

| 항목 | 가중치 | 수식 (활성화 시) | 설명 |
|------|--------|------------------|------|
| `flat_orientation_l2` | 0 | $g_s \cdot (g_{b,x}^2 + g_{b,y}^2)$ | 수평 자세(기울기) L2 패널티 |
| `base_height_l2` | 0 | $g_s \cdot (z_w - h_{\text{target}}^*)^2$ | 목표 높이 유지 ($h_{\text{target}}=0.40$ m) |
| `body_lin_acc_l2` | 0 | $\sum_j \|\mathbf{a}_j\|_2$ | 베이스 선가속도 L2 패널티 |

$h_{\text{target}}^* = h_{\text{target}} + \text{mean}(\text{height\_scanner\_base})$

---

## 3. 관절(Joint) 안정성

다리 관절(12개) 및 바퀴 관절(4개)의 부드러운 동작, 한계 준수, 대칭성을 다룹니다.

**다리 관절**: `FR/FL/RR/RL_(hip|thigh|calf)_joint`  
**바퀴 관절**: `FR/FL/RR/RL_foot_joint`

### 3.1 `joint_torques_l2` — 관절 토크 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-2.5 × 10⁻⁵** |
| 대상 | 다리 관절 12개 |
| 구현 | `isaaclab.envs.mdp.rewards` |

$$
r_{\text{joint\_torques}}^{\text{raw}} = \sum_{j \in \mathcal{J}_{\text{leg}}} \tau_j^2
$$

- $\tau_j$: 관절 $j$의 적용 토크 (`applied_torque`)

---

### 3.2 `joint_acc_l2` — 다리 관절 가속도 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-2.5 × 10⁻⁷** |
| 대상 | 다리 관절 12개 |

$$
r_{\text{joint\_acc}}^{\text{raw}} = \sum_{j \in \mathcal{J}_{\text{leg}}} \ddot{q}_j^2
$$

- $\ddot{q}_j$: 관절 가속도 (`joint_acc`)

---

### 3.3 `joint_acc_wheel_l2` — 바퀴 관절 가속도 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-2.5 × 10⁻⁹** |
| 대상 | 바퀴 관절 4개 |

$$
r_{\text{joint\_acc\_wheel}}^{\text{raw}} = \sum_{j \in \mathcal{J}_{\text{wheel}}} \ddot{q}_j^2
$$

---

### 3.4 `joint_pos_limits` — 관절 위치 한계 위반 패널티

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-5.0** |
| 대상 | 다리 관절 12개 |

$$
r_{\text{joint\_pos\_limits}}^{\text{raw}} = \sum_{j \in \mathcal{J}_{\text{leg}}} \left[ \max(0,\ q_{\min,j} - q_j) + \max(0,\ q_j - q_{\max,j}) \right]
$$

- $q_j$: 관절 위치, $[q_{\min,j},\ q_{\max,j}]$: 소프트 한계 (`soft_joint_pos_limits`)

---

### 3.5 `joint_power` — 관절 전력(일률) 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-2.0 × 10⁻⁵** |
| 대상 | 다리 관절 12개 |

$$
r_{\text{joint\_power}}^{\text{raw}} = \sum_{j \in \mathcal{J}_{\text{leg}}} \left|\dot{q}_j \cdot \tau_j\right|
$$

---

### 3.6 `stand_still` — 정지 명령 시 자세 유지

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-2.0** |
| 대상 | 다리 관절 12개 |
| 파라미터 | `command_threshold = 0.1` |

$$
r_{\text{stand\_still}}^{\text{raw}} = g_s \cdot \mathbb{1}\left[\|\mathbf{c}\| < c_{\text{th}}\right] \cdot \sum_{j \in \mathcal{J}_{\text{leg}}} \left|q_j - q_j^{\text{default}}\right|
$$

- $\mathbf{c}$: 속도 명령 (`base_velocity`), $c_{\text{th}} = 0.1$
- $q_j^{\text{default}}$: 기본 관절 위치
- **목적**: 명령이 거의 0일 때 기본 자세에서 벗어나지 않도록 유도

---

### 3.7 `joint_pos_penalty` — 관절 위치 편차 패널티

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-1.0** |
| 대상 | 다리 관절 12개 |
| 파라미터 | `stand_still_scale=5.0`, `velocity_threshold=0.5`, `command_threshold=0.1` |

$$
\delta_q = \left\| \mathbf{q}_{\text{leg}} - \mathbf{q}_{\text{leg}}^{\text{default}} \right\|_2
$$

$$
r_{\text{joint\_pos\_penalty}}^{\text{raw}} = g_s \cdot
\begin{cases}
\delta_q & \text{if } \|\mathbf{c}\| > c_{\text{th}} \text{ or } \|\mathbf{v}_{b,xy}\| > v_{\text{th}} \\
5 \cdot \delta_q & \text{otherwise}
\end{cases}
$$

- $\mathbf{v}_{b,xy}$: 베이스 xy 평면 선속도, $v_{\text{th}} = 0.5$
- **목적**: 이동 중에는 관절 편차를 완화하고, 정지/저속 시에는 더 강하게 기본 자세 유지

---

### 3.8 `joint_mirror` — 좌우 대칭 관절 패턴

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-0.05** |
| 대상 | 다리 관절 (대칭 쌍) |

대칭 쌍:
- FR ↔ RL: `(hip, thigh, calf)`
- FL ↔ RR: `(hip, thigh, calf)`

$$
r_{\text{joint\_mirror}}^{\text{raw}} = g_s \cdot \frac{1}{N_{\text{pairs}}} \sum_{k=1}^{N_{\text{pairs}}} \left(q_{\text{left},k} - q_{\text{right},k}\right)^2
$$

- $N_{\text{pairs}} = 2$
- **목적**: 대칭 보행/자세로 안정성 향상

---

## 4. 행동(Action) 안정성

### 4.1 `action_rate_l2` — 행동 변화율 억제

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-0.01** |
| 대상 | 전체 행동 벡터 (다리 위치 + 바퀴 속도) |
| 구현 | `isaaclab.envs.mdp.rewards` |

$$
r_{\text{action\_rate}}^{\text{raw}} = \left\| \mathbf{a}_t - \mathbf{a}_{t-1} \right\|_2^2
$$

- $\mathbf{a}_t$: 현재 스텝 행동 (`action_manager.action`)
- $\mathbf{a}_{t-1}$: 이전 스텝 행동 (`action_manager.prev_action`)
- **목적**: 급격한 제어 입력 억제 → 진동·불안정 방지

---

## 5. 접촉(Contact) 안정성

### 5.1 `undesired_contacts` — 원치 않는 접촉 패널티

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-1.0** |
| 대상 | 발(foot) 이외 모든 바디 |
| 파라미터 | `threshold = 1.0` N |

$$
r_{\text{undesired\_contacts}}^{\text{raw}} = g_s \cdot \sum_{b \in \mathcal{B}_{\text{undesired}}} \mathbb{1}\left[\max_t \|\mathbf{f}_{b,t}\| > f_{\text{th}}\right]
$$

- $\mathcal{B}_{\text{undesired}}$: `^(?!.*_foot).*` 패턴에 매칭되는 바디 (대퇴, 몸체 등)
- $\mathbf{f}_{b,t}$: 접촉력 이력 (`net_forces_w_history`), $f_{\text{th}} = 1.0$
- **목적**: 다리/몸체가 지면·장애물에 닿지 않도록 유도

---

### 5.2 `contact_forces` — 과도한 발 접촉력 패널티

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **-1.5 × 10⁻⁴** |
| 대상 | 발 링크 (`.*_foot`) |
| 파라미터 | `threshold = 100.0` N |

$$
r_{\text{contact\_forces}}^{\text{raw}} = \sum_{b \in \mathcal{B}_{\text{foot}}} \max\left(0,\ \max_t \|\mathbf{f}_{b,t}\| - f_{\text{th}}\right)
$$

- $f_{\text{th}} = 100.0$ N
- **목적**: 발에 가해지는 충격 하중 완화

---

### 5.3 `feet_contact_without_cmd` — 무명령 시 발 접촉 보상

| 항목 | 값 |
|------|-----|
| 가중치 $w$ | **+0.1** |
| 대상 | 발 링크 4개 |

$$
r_{\text{feet\_contact\_wo\_cmd}}^{\text{raw}} = g_s \cdot \mathbb{1}\left[\|\mathbf{c}\| < 0.1\right] \cdot \sum_{b \in \mathcal{B}_{\text{foot}}} \mathbb{1}[\text{first\_contact}_b]
$$

- **목적**: 속도 명령이 없을 때 발이 지면에 닿아 있도록 유도 (정지 안정성)

---

### 5.4 비활성 접촉·보행 안정성 항목 (가중치 0)

| 항목 | 가중치 | 설명 |
|------|--------|------|
| `feet_stumble` | 0 | 수평 접촉력 > 4×수직 접촉력 시 걸림 패널티 |
| `feet_slide` | 0 | 접촉 중 발 미끄러짐(측면 속도) 패널티 |
| `feet_air_time` | 0 | 보행 시 발 공중 시간 보상 |
| `feet_contact` | 0 | 기대 접촉 수(2) 불일치 패널티 |
| `feet_height` / `feet_height_body` | 0 | 발 높이 제어 |
| `feet_gait` | 0 | 대각 보행(trot) 동기화 보상 |

---

## 6. 활성 안정성 보상 요약표

| # | 항목 | 가중치 $w$ | 유형 | 핵심 역할 |
|---|------|-----------|------|-----------|
| 1 | `upward` | +1.0 | 보상 | 똑바로 선 자세 유지 |
| 2 | `feet_contact_without_cmd` | +0.1 | 보상 | 무명령 시 발 접지 |
| 3 | `lin_vel_z_l2` | -2.0 | 패널티 | 수직 바운스 억제 |
| 4 | `stand_still` | -2.0 | 패널티 | 정지 시 기본 자세 유지 |
| 5 | `joint_pos_limits` | -5.0 | 패널티 | 관절 한계 준수 |
| 6 | `joint_pos_penalty` | -1.0 | 패널티 | 관절 편차 억제 |
| 7 | `undesired_contacts` | -1.0 | 패널티 | 몸체/다리 비정상 접촉 방지 |
| 8 | `ang_vel_xy_l2` | -0.05 | 패널티 | 롤/피치 각속도 억제 |
| 9 | `joint_mirror` | -0.05 | 패널티 | 좌우 대칭 자세 |
| 10 | `action_rate_l2` | -0.01 | 패널티 | 행동 급변 억제 |
| 11 | `joint_torques_l2` | -2.5e-5 | 패널티 | 토크 최소화 |
| 12 | `joint_power` | -2.0e-5 | 패널티 | 관절 일률 최소화 |
| 13 | `contact_forces` | -1.5e-4 | 패널티 | 과도한 발 충격 억제 |
| 14 | `joint_acc_l2` | -2.5e-7 | 패널티 | 다리 관절 가속도 억제 |
| 15 | `joint_acc_wheel_l2` | -2.5e-9 | 패널티 | 바퀴 관절 가속도 억제 |

---

## 7. 속도 추적 보상 (안정성 외 — 태스크 목표)

아래 항목은 **안정성**보다 **명령 속도 추적**이 목적이지만, 추적 보상에도 $g_s$가 곱해져 기울어진 상태에서는 추적 보상이 감소합니다.

| 항목 | 가중치 | 수식 |
|------|--------|------|
| `track_lin_vel_xy_exp` | +3.0 | $g_s \cdot \exp\!\left(-\dfrac{\|\mathbf{c}_{xy} - \mathbf{v}_{b,xy}\|^2}{\sigma^2}\right)$, $\sigma = 0.5$ |
| `track_ang_vel_z_exp` | +1.5 | $g_s \cdot \exp\!\left(-\dfrac{(c_z - \omega_{b,z})^2}{\sigma^2}\right)$, $\sigma = 0.5$ |

---

## 8. 기호 정리

| 기호 | 의미 |
|------|------|
| $\mathbf{c}$ | 속도 명령 벡터 `[v_x^*, v_y^*, \omega_z^*]` |
| $\mathbf{v}_b$ | 베이스 좌표계 선속도 |
| $\boldsymbol{\omega}_b$ | 베이스 좌표계 각속도 |
| $g_b$ | 베이스 좌표계 투영 중력 |
| $g_s$ | 중력 정렬 스케일링 계수 |
| $q_j,\ \dot{q}_j,\ \ddot{q}_j$ | 관절 위치, 속도, 가속도 |
| $\tau_j$ | 관절 토크 |
| $\mathbf{a}_t$ | 정책 행동 벡터 |
| $\mathbb{1}[\cdot]$ | 지시 함수 (조건 만족 시 1, 아니면 0) |

---

## 9. 소스 파일 참조

| 파일 | 경로 |
|------|------|
| 환경 설정 | `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py` |
| 보상 함수 (Robot Lab) | `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py` |
| 보상 함수 (Isaac Lab) | `isaaclab/envs/mdp/rewards.py` |
| 보상 템플릿 | `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` |
