## 실제 환경에서의 고속 비행 학습

### 0. 논문 정보 (Reference)
* **Title:** Learning High-Speed Flight in the Wild
* **Authors:** Antonio Loquercio, Elia Kaufmann, René Ranftl, Matthias Müller, Vladlen Koltun, Davide Scaramuzza
* **Journal:** Science Robotics Vol. 6, Issue 59, 2021
* **DOI:** 10.1126/scirobotics.abg5810
* **arXiv:** 2110.05113
* **Open Source:** https://github.com/uzh-rpg/agile_autonomy

---

### 1. Introduction

쿼드로터는 숲이나 도시 협곡처럼 복잡한 환경을 고속으로 통과할 수 있지만 이를 활용하려면 숙련된 조종사가 필요하다. 기존 자율 비행 시스템은 대체로 감지 → 맵핑 → 경로계획 → 제어의 단계적 파이프라인에 의존한다. 이 구조는 모듈화와 해석 가능성 측면에서는 장점이 있지만, 각 단계의 latency가 누적되고 오차가 전파되며 모듈 간 상호작용이 충분히 모델링되지 않아 고속 비행에서 근본적인 제약이 된다. 특히 맵핑 기반 접근은 여러 관측을 통합해야 하므로 처리 지연이 커지고 고속 이동 시에는 관측 간 중첩이 줄어 맵의 품질도 함께 저하된다.

기존 end-to-end 비행 정책도 존재했지만, 많은 경우 평면 운동이나 이산 행동 등으로 문제를 단순화하거나 장애물이 없는 자유공간에서의 민첩한 제어에 초점을 맞추었다. 이에 비해 본 논문은 stereo-derived depth를 중간 표현으로 사용해 sim-to-real gap을 줄이고, privileged expert를 모방하는 다중 가설 정책을 시뮬레이션에서만 학습한 뒤 실제 복잡 환경에 zero-shot으로 전이한다. 또한 정책 네트워크를 경량화하여 온보드 실시간 추론이 가능하도록 설계하였다.

---

### 2. Results

본 논문의 실험 결과는 실환경 고속 비행, 제어된 시뮬레이션, 계산 복잡도 비교, 센서 지연 및 노이즈 영향의 네 가지 하위 섹션으로 구성된다.

#### A. High-Speed Flight in the Wild

실험은 자연 환경(숲, 산악, 설원)과 인간 환경(무너진 건물, 비행기 격납고, 도시 거리 등)에서 총 56회의 실험을 수행하였다. 모든 실험에서 길이 40m의 직선 또는 반지름 6m의 원형 reference trajectory가 주어졌으며, 이는 그대로 추종하면 장애물과 충돌할 수 있는 경로였다. 드론은 이 reference를 장기 목표로만 사용하고, 실제 비행에서는 장애물을 감지할 때마다 즉석에서 회피해야 했다. 성능은 목표 지점 반경 5m 이내에 충돌 없이 도달했는지로 평가한다.

자연 환경에서의 실패는 주로 물체가 시야에 매우 늦게 들어오거나, 폭 1m 미만의 매우 좁은 구멍 근처에서 한 번의 잘못된 조향이 곧바로 충돌로 이어질 때 발생했다. 인간 환경은 장애물의 크기와 형상이 매우 다양하고, 멀리 있는 구멍을 향해 더 이른 시점부터 회피를 시작해야 한다는 점에서 자연 환경과는 다른 난점을 가진다.

실환경 검증에 관하여 해당 정책은 시뮬레이션에서만 학습되었으며, 실제 자연 및 인공 환경에는 추가 fine-tuning 없이 zero-shot으로 배치되었다. 총 56회 실험에서 전체 누적 성공률은 3, 5m/s에서 100%, 7m/s에서 80%, 10m/s에서 60%였다. 자연 환경(31회)에서는 40m 직선 혹은 반지름 6m 원형 참조 궤적을 따라 숲, 산악,설원에서 검증했으며, 7m/s에서 80% 성공, 10m/s에서 60% 성공을 기록했다. 인간 환경(19회)에서는 3~7m/s 범위에서 모두 충돌 없이 목표에 도달했다. 별도 비교에서 상용 드론 Skydio R1은 약 0.8m 개구 통과 과제에서 3회 모두 실패한 반면, 제안 방법은 3m/s, 5m/s에서 6회 모두 성공했다.

---

#### B. Controlled Experiments

FastPlanner와 Reactive baseline은 모두 동일하게 플랫폼 state, stereo depth, 그리고 5 s 앞의 reference state를 목표로, Blind는 reference만 추종하는 난이도 기준선이다. 시뮬레이션은 Flightmare + RotorS Gazebo + Unity에서 수행되었고, 숲, 좁은 틈, 재난, 도시 네 환경에서 평가했다. 저속(3m/s)에서는 성능 차이가 작지만, 속도가 증가하면 baseline은 급격히 붕괴하고 제안 방법은 10m/s에서 평균 70% 성공률을 유지한다. 특히 재난, 도시 환경에서는 조기에 free space 방향으로 steering하면서도 장애물 근처에서는 미세한 reactive correction을 수행하는 능력을 갖는다. 

---

#### C. Computational Cost

세 방법의 구성별 처리 지연을 측정한 결과는 다음과 같다.

| Method | Pre-processing (ms) | Mapping/NN (ms) | Projection (ms) | Total Latency (ms) | 비고 |
|---|---:|---:|---:|---:|---|
| FastPlanner | 14.6 ± 2.3 | 49.2 ± 8.7 (mapping) | 1.4 ± 1.6 | 65.2 | 여러 관측을 통합해야 함. |
| Reactive | 13.8 ± 1.3 | 5.3 ± 0.9 (primitive planning) | – | 19.1 | 지연은 적지만 표현력이 제한적. |
| Ours (desktop CPU) | 0.1 ± 0.04 | 10.1 ± 1.5 (NN inference) | 0.08 ± 0.01 | 10.3 | GPU 사용 시 2.6ms로 단축. |
| Ours (Jetson TX2) | 0.2 ± 0.1 | 38.9 ± 4.5 | 2.5 ± 1.5 | 41.6 | 온보드 PC 성능 기준. |

표의 비교는 공정성을 위해 우선 모든 방법을 desktop CPU 기준으로 맞춰 측정하고 제안 방법에 한해 GPU 추론 시간과 Jetson TX2 온보드 시간을 추가로 함께 제시한다. 여기서 baseline의 pre-processing은 depth map으로부터 point cloud를 만드는 시간이고 제안 방법의 pre-processing은 depth를 neural network 입력 tensor로 변환하는 시간이다. CPU만 사용해도 제안 방법은 baseline보다 낮은 지연을 보이며, GPU 사용 시 총 latency는 2.6ms 수준까지 줄어든다. 온보드 기준 총 latency는 41.6ms로 약 24 Hz 업데이트가 가능하다.

---

#### D. The effect of latency and sensor noise

해당 제어 연구는 단일 장애물 회피 과제를 사용한다. 쿼드로터가 일정한 전방 속도로 직진하면서 제한된 센서 범위 안에서 하나의 pole을 측면으로 회피해야 하는 설정이다. 전방 속도는 3~13m/s 범위에서 변화시키고, ground-truth depth와 stereo-estimated depth 조건을 나누어 latency와 noise의 영향을 비교한다.

관련 이론 분석에 대하여 Falanga et al.의 pole avoidance setting을 확장하여, sensing range, sensor latency, processing latency뿐 아니라 회피에 필요한 rotational delay까지 반영한 theoretical 최대 속도를 계산했다. Ground-truth depth에서는 모든 방법이 5m/s까지는 성공했지만, Reactive는 공격적인 primitive 부족으로, FastPlanner는 sub-optimal action 때문에 그 이상에서 급격히 악화되었다. Stereo depth에서는 FastPlanner가 outlier rejection을 위해 2~3회 observation을 필요로 하며 5m/s 이상에서 사실상 실패했고, Reactive도 current observation 의존성 때문에 7m/s에서 유의미한 성능 저하를 보였다. 반면 제안 방법은 theoretical limit에 가장 가깝게 동작했으며, 10m/s에서도 약 10% 수준의 성능 저하만 나타났다.

---

### 3. Discussion

기존 자율 비행 시스템은 모듈화를 통해 구현을 단순화하였지만 모듈간 지연과 오차 누적 및 상호작용 부재가 고속 비행에 근본적인 제약을 준다. 이에 따라 해당 연구는 감지, 맵핑, 경로계획을 통합한 단일 신경망 함수로 대체하였다. 이 신경망은 depth 이미지, 속도, 자세, 목표 방향을 입력으로 받아 충돌 위험이 낮은 세 가지 궤적을 예측하고 비용 최소 궤적을 MPC로 추종한다. 이 방법은 모듈 간 통신 없이 한 번의 추론으로 경로를 산출하므로 지연이 크게 줄고 데이터 기반으로 센서 노이즈에 대한 견고성을 갖는다.

이에 따라 성공 요인을 3가지로 나눌 수 있다.
1) 기술적 기여를 위한 전문가 모방학습으로 복잡한 다봉 분포를 효율적으로 학습한다.
2) 다중 모드 출력을 통해 여러 회피 전략을 동시에 고려한다.
3) 깊이 기반 추상 입력으로 시뮬레이션-현실 차이를 최소화한다.
이러한 설계 조합이 고속 비행에서 기존 방법 대비 10배 이상 낮은 실패율을 달성하게 했다.

한계점으론 10m/s 이상에서는 단지 센서 한계 때문만이 아니라, 장시간의 temporal consistency와 장애물 밀도에 따른 속도 조절을 동시에 만족하는 feasible trajectory 자체가 매우 희소해져 sampling 기반 전문가도 해를 찾기 어렵다. 여기에 공력, motor delay, battery voltage drop에 따른 dynamics mismatch와 perception latency가 겹치며 실제 성능이 더 떨어진다. 따라서 향후에는 더 높은 fidelity의 시뮬레이션, model mismatch에 강한 학습, event camera 기반 저지연 perception, 그리고 장기 planner와의 결합이 중요하다.

---

### 4. Materials and Methods

에이전트의 관측은 depth image, 플랫폼 속도, 자세, 목표를 나타내는 desired direction으로 구성되며, 출력은 1초 horizon의 trajectory hypotheses와 각 궤적의 collision risk이다. 상위 planner 혹은 사용자가 제공하는 reference trajectory는 collision-free일 필요가 없고, 학생 정책은 이를 장기 목표로만 사용하면서 단기 장애물 회피를 담당한다. 학습은 전적으로 simulation에서 수행되며 sim-to-real gap을 줄이기 위해 simulated stereo pair에서 SGM으로 생성한 depth를 사용한다.

#### A. Privileged Expert

Preivileged Expert는 단일 해를 직접 푸는 최적화기가 아니라, 완전한 3D 환경 지도와 정확한 상태를 이용해 $P(\tau \mid \tau_{\mathrm{ref}}, C)$ 로 표현되는 collision-free trajectory distribution을 샘플링하는 offline planner다. Metropolis–Hastings를 이용해 약 50K개의 후보 궤적을 생성하고, cubic B-spline 제어점을 구면좌표계에서 샘플링해 동역학적으로 추종 가능한 궤적을 만든다. 또한 raw reference 대신 start-to-goal의 global collision-free trajectory $\tau_{\mathrm{gbl}}$ 주변으로 샘플링을 bias하여 탐색 범위를 줄이고 더 보수적인 회피 궤적을 얻는다. 최종적으로 충돌 궤적을 제거한 뒤 비용이 낮은 상위 3개만 학생 정책의 supervision으로 사용한다.

$$P(\tau \mid \tau_{\mathrm{ref}}, C)=\frac{1}{Z}\exp\bigl(-c(\tau,\tau_{\mathrm{ref}},C)\bigr)$$

$$c(\tau,\tau_{\mathrm{ref}},C)=\int_{0}^{1}\lambda_c\,C_{\mathrm{collision}}(\tau(t))\,dt+\int_{0}^{1}\bigl[\tau(t)-\tau_{\mathrm{ref}}(t)\bigr]^{\top}Q\bigl[\tau(t)-\tau_{\mathrm{ref}}(t)\bigr]\,dt$$

여기서 λ_c=1000은 충돌 비용 가중치, Q는 상태 오차 가중 행렬이다. C_collision은 쿼드로터를 반경 0.2m의 구로 모델링하여 장애물까지의 거리에 따른 비용을 정의한다. 거리가 $2r_q$보다 크면 0이고, 그 이하에서는 $−d_c^2/r_q^2 + 4$로 증가한다.

분포 P는 환경에 따라 multi-modal 구조를 가지므로, 전문가 알고리즘은 Metropolis–Hastings 샘플링을 통해 약 5만 개의 궤적을 샘플링한다. 궤적은 3개의 제어점을 갖는 cubic B-spline으로 표현하고, 제어점을 구면 좌표계에서 샘플링하여 동역학적 연속성(위치, 속도, 가속도)을 유지한다. 또한 글로벌 최적 경로 $\tau_{gbl}$을 미리 계산하여 샘플링을 해당 궤적 주변으로 유도함으로써 탐색 공간을 줄이고 보수적인 궤적을 생성한다. 충돌한 궤적은 제거하고 비용이 낮은 상위 3개 궤적만 학생 정책의 학습 레이블로 사용한다.

---

#### B. The Student Policy

학생 정책은 SGM depth, velocity, attitude와 가장 가까운 reference state 기준 1초 뒤 reference point를 향하는 desired direction을 입력으로 받아, 1초 horizon의 위치 궤적 3개와 각 궤적의 relative collision cost를 예측한다. 여기서 다중 모드 출력이 중요한 이유는 obstacle avoidance가 본질적으로 multi-modal이므로 가능한 회피 궤적들의 평균이 오히려 충돌 경로가 되기 쉽기 때문이다. 학습에는 R-WTA 손실을 사용해 각 expert trajectory를 가장 가까운 network hypothesis에 주로 매칭하고 projection 단계에서는 5차 다항식과 평균 속도 제약을 이용해 급격한 pitching 없이 연속적인 추종이 가능하도록 만든다.

학생 정책은 온보드 센서만을 사용하여 실시간으로 궤적을 생성한다.
입력은 depth 이미지 d ∈ ℝ^{640×480}, 플랫폼 속도 v ∈ ℝ³, 자세 q ∈ ℝ⁹, 참조 방향 ω ∈ ℝ³이며, 출력은 M=3개의 궤적 후보와 각 후보의 충돌 비용이다.

이에 대한 네트워크는 두 branch로 구성된다.
- Visual branch: MobileNet-V3 백본과 1D convolution을 통해 깊이 영상에서 모드당 32차원의 특징을 추출한다.
- State branch: 속도, 자세, 참조 방향을 네 층의 MLP([64, 32, 32, 32])로 처리하여 32차원 특징을 얻는다. 이후 두 특징을 모드별로 결합하여 4층 MLP([64, 128, 128])를 통해 궤적과 비용을 출력한다. 예측된 궤적 $τ_kn$은 0.1 s 간격으로 10개의 위치 샘플을 포함하며, 제어점 대신 직접 위치를 출력하여 테스트 시 보간 비용을 줄였다.

각 샘플에 대해 전문가가 제공한 3개의 궤적 Te와 네트워크가 예측한 3개의 궤적 Tn의 대응관계를 Relaxed Winner-Takes-All(R-WTA) 손실로 최적화한다. 이 손실은 가장 가까운 네트워크 궤적에는 가중치 0.95, 나머지에는 0.025를 부여하여 모드 붕괴를 방지한다. 또한 각 궤적의 충돌 비용 예측 $c_k$는 실제 충돌 비용 C_collision(τ_kn)과 L2 손실로 학습한다. 최종 손실은 두 항을 λ₁=10, λ₂=0.1의 비율로 가중합하여 Adam optimizer로 학습한다.

네트워크가 출력하는 위치 샘플은 5차 다항식으로 투영하여 위치, 속도, 가속도의 연속성을 만족하도록 변환한다. 이때 각 축에 대해 계수 $a_x$를 최소 제곱으로 찾고 현재 상태와의 연속 조건을 부과한다. 투영된 궤적 중에서 비용이 최소 값의 95% 이내인 후보들 중 입력 비용이 가장 작은 것을 선택해 MPC로 추종한다.(이는 연속성을 유지하여 불필요한 방향 전환을 피한다.)

---

#### C. Training Environments

Flightmare와 Unity 엔진을 이용해 총 850개의 훈련 환경을 생성했다. 훈련 환경은 두 범주로 나뉜다.

- 나무가 있는 숲: 다양한 높이·두께의 나무를 Poisson 점 과정에 따라 배치했다.
- 볼록 형태들: 타원체, 상자, 실린더 등 볼록 도형을 랜덤 크기(0.5~4m, 높이 0.5~8m)로 배치했다.
모든 환경에서 intensity δ를 4~7 사이에서 랜덤으로 샘플링하여 장애물 밀도를 다양화했다.
각 환경마다 시작 지점에서 40m 앞까지의 글로벌 경로 $τ_gbl$을 계산하고, 학생 정책은 직선 참조 궤적만 받아 목표를 인식한다.

DAgger를 사용할 때는 초기 학습 단계의 불안정을 막기 위해, 드론이 global trajectory ​$\tau_{gbl}$에서 일정 거리 이상 벗어나면 student prediction 대신 $\tau_{gbl}$을 MPC로 직접 추종한다. 이 threshold $\xi$는 0에서 시작해 30개 환경마다 0.25씩 증가시키며 최대 6까지 확장한다. 전체 환경에서 약 90K 샘플을 수집했고, narrow-gap 과제를 위해 100개의 추가 환경에서 약 10K 샘플로 fine-tuning했다. 이 narrow-gap fine-tuning 환경은 길이 50m의 벽과 중앙의 단일 vertical gap으로 구성된다. gap의 폭은 0.7~1.2m 범위에서 무작위로 정해지며, 드론은 벽으로부터 10m 떨어진 지점에서 시작하고 gap에 대한 lateral offset도 무작위로 부여된다.배치 시에는 시뮬레이션에서 simulated stereo + SGM + ground-truth state를 사용하고, 실제 기체에서는 RealSense D435와 T265를 사용한다.

---

#### D. Method Validation

설계 선택의 중요성을 검증하기 위해 ablation study를 수행했다. 핵심 ablation은 (1) 전문가 샘플링의 global-plan initialization 제거(LocalPlan), (2) depth intermediate representation을 RGB로 대체(RGB), (3) multi-modal prediction을 단일 모드로 제한(UniModal)하는 것이며, 여기에 출력 representation을 1s trajectory 대신 single waypoint로 축소한 비교(SinglePoint)를 추가했다. 결과적으로 네 설계 모두 성능에 영향을 주었고, depth 표현과 multi-modal prediction의 부재는 고속에서 큰 성능 저하를 유발했다. UniModal이 약한 이유는 회피 문제가 본질적으로 다봉적이어서 여러 expert trajectory의 평균이 오히려 충돌 경로가 되기 쉽기 때문이다. 또한 global-plan initialization은 저속에서는 영향이 작지만, 고속에서는 obstacle-dense region을 미리 피하도록 bias를 주어 성공률을 높인다.

- Global planning 초기화(LocalPlan): 전문가 샘플링에서 글로벌 경로 초기화를 제거하면 고속에서 성능이 급격히 떨어진다. 다중 모드 학습(UniModal): 모드 하나만 예측하는 모델은 환경의 다모드성을 처리하지 못해 성능이 크게 감소한다.
- RGB 입력(RGB): 깊이 대신 컬러 이미지로 학습하면 도메인 갭이 커져 성능이 하락한다.
- 단일 포인트 예측(SinglePoint): 1s 전체 궤적 예측 대비 표현력이 부족해 고속에서 성능이 크게 떨어졌다.

---

### 5. Acknowledgments & Supplementary Materials

추가 참고자료.(6개의 보조 섹션과 여러 그림/영상 자료)

S1. 추정 및 제어 노이즈에 대한 민감도 분석
S2. 장애물 밀도 변화가 성능에 미치는 영향
S3. 실험 플랫폼 하드웨어 세부 사항
S4. 계산 복잡도 세부 분석
S5. 단일 폴 회피 시 이론적 최대 속도 산출
S6. Metropolis–Hastings 샘플링 기법
Figure S1–S6, Movie S1–S2 포함.

---

### 6. Conclusion

본 연구는 고속 자율 비행을 위해 기존의 모듈화된 파이프라인을 깨고, 특권 모방 학습 + 다중 모드 예측 + MPC라는 하이브리드 구조를 제시하였다. 중요한 점은 모델 전부가 완전한 end-to-end가 아니라, 신경망이 단기 궤적 후보를 제안하고 MPC가 동역학적으로 추종함으로써 안정성과 제약 만족을 보장한다는 것이다.

'왜 설계가 합리적인가?'를 한 번 생각해보면 다음과 같다.
1) 맵을 유지하지 않으므로 지연이 현저히 줄고 고속 환경에서 치명적인 센서 지연 문제를 완화한다.
2) 다중 모드 예측과 R-WTA 손실은 회피 경로의 다봉성을 포착해 mode collapse를 방지한다.
3) depth 입력 사용은 시뮬레이션과 현실 사이의 도메인 갭을 줄여 zero-shot 전이를 가능하게 한다.
4) 학생 정책은 경량 구조로 온보드 임베디드 시스템에서도 실시간으로 실행 가능하며, GPU 이용 시 지연이 수 밀리초 수준이다.

해당 연구는 탐사, 구조, 배달 등에서 고속 자율 드론의 실용화 가능성을 크게 넓힌다. 다만 현존하는 한계점으론 실제 환경에서 10m/s 이상은 여전히 어려우며, 단기 회피 정책과 장기 경로 계획의 통합에 대한 논의가 필요하다. 따라서 전반적인 공학적 의의는 ‘완성된 범용 해법'보다는, 현 상태에서 실용적으로 다루는 '저지연 및 고기동 자율 비행'을 위한 설계 원칙을 제시했다는데에 의의가 있다.

---

**Review by 변정우, Aerospace Engineering Undergraduate Researcher**
**[Update - Time Log]**
* 2026.02.28: [Draft] 전체적인 내용 리딩 완료 및 초안 작성
* 2026.03.10: [ver_1] 원문과 비교하여 전반적인 내용 업데이트
* 2026.03.11: [ver_2] 세부적인 내용 추가, 내용 정리하여 리뷰 정리
* 2026.03.17: [Final_ver] 전반적인 리뷰 작업 및 결론 논의 검토
