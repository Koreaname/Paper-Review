## Tight Constraint Prediction of Six-Degree-of-Freedom Transformer-based Powered Descent Guidance

### 0. 논문 정보 (Reference)
* Title: Tight Constraint Prediction of Six-Degree-of-Freedom Transformer-based Powered Descent Guidance  
* Authors: Julia Briden, Trey Gurga, Breanna Johnson, Abhishek Cauligi, Richard Linares  
* arXiv: 2501.00930  
* Category: Optimization / Optimal Control / Guidance & Control / Learning-based Optimization  
* Affiliation: NASA Johnson Space Center, MIT, NASA Jet Propulsion Laboratory  

### 1. Nomenclature

### Variables

이 논문은 6-DoF powered descent guidance 문제를 **상태, 제어, 제약, 최적화 보조변수**의 네 층위로 다룬다. 뒤의 수식을 읽을 때 핵심이 되는 기호는 아래와 같다.

$$
x(t)=\begin{bmatrix}
r_I(t) & v_I(t) & q_{B\leftarrow I}(t) & \omega_B(t) & m(t)
\end{bmatrix}^\top,\qquad
u(t)=T_B(t)
$$

* $r_I$, $v_I$는 관성좌표계에서의 위치와 속도다.
* $q_{B\leftarrow I}$는 관성좌표계에서 body frame으로의 자세를 나타내는 quaternion이고, $\omega_B$는 body frame 각속도다.
* $m$은 질량, $T_B$는 body frame 추력 벡터다.
* $A_B$는 body frame 공력, $M_B=r_{T,B}\times T_B+r_{cp,B}\times A_B$는 추력과 공력으로 생기는 총 토크다.
* $t_f$는 종단시간, $N$은 이산화 노드 수, $\sigma_i$는 time-scaling factor다.
* $\tau_i$는 샘플 $i$에서 **tight constraint set**, 즉 최적해에서 실제로 경계에 걸리는 제약들의 집합이다.
* $\theta_i$는 샘플 $i$의 신경망 입력 파라미터이고, $s_i$는 신경망이 출력하는 전략(strategy)이다.
* $\nu$는 SCvx에서 사용하는 virtual control, $\omega$는 penalty coefficient다.
* $T_{\min}, T_{\max}, \delta_{\max}, \theta_{\max}, \omega_{\max}, m_{\mathrm{dry}}$는 각각 추력 하한/상한, 짐벌 한계, 기체 pointing 한계, 각속도 한계, 건조질량 하한을 뜻한다.

### Functions

논문은 일반적인 비선형 최적화 문제를 다음과 같이 표기한다.

$$
f:\mathbb{R}^n\rightarrow \mathbb{R}
$$

* $f(z)$는 목적함수다.
* $g(z)$는 비볼록 equality/inequality constraints를 모은 함수다.
* $h(z)$는 convex inequality constraints를 모은 함수다.

즉, 이후의 6-DoF PDG도 결국은 **비볼록 동역학 + 비볼록 상태/제어 제약 + 일부 볼록 제약**으로 구성된 매개변수화된 최적화 문제로 이해하면 된다.

### Notation

* $\otimes$는 quaternion multiplication이다.
* $\Omega$는 quaternion kinematics를 쓰기 위해 도입한 skew-symmetric matrix다.
* $\hat e$는 단위 방향 벡터다.
* $I_{\mathrm{eq}}$, $I_{\mathrm{ineq}}$, $J_{\mathrm{ineq}}$는 각각 비볼록 equality, 비볼록 inequality, convex inequality의 인덱스 집합이다.
* $q_{B\leftarrow I}^\ast$는 conjugate quaternion, $q_{\mathrm{id}}$는 identity quaternion이다.

이 표기 체계의 핵심은 간단하다. **SCvx는 이 비볼록 문제를 반복적으로 선형화하고, T-SCvx는 그 과정에서 어떤 inequality가 실제로 경계에 걸리는지 예측해 계산량을 줄인다.**

---

### 2. Introduction

행성 착륙 문제에서 powered descent guidance(PDG)는 단순히 “착륙 가능한 궤적”을 찾는 수준을 넘어서, **연료를 최소화하면서도 동역학적으로 실현 가능하고 안전 제약을 만족하는 궤적**을 실시간에 가깝게 생성해야 한다. 특히 인간 탑승 가능급의 고질량 임무로 갈수록 연료 여유와 안전 마진이 임무 성패를 가르기 때문에, onboard에서 long-horizon trajectory generation을 수행할 수 있는 계산 효율이 중요해진다.

하지만 실제 우주비행체에 쓰이는 radiation-hardened processor는 일반적인 지상용 GPU/CPU보다 계산 자원이 훨씬 제한적이다. 6-DoF 착륙 문제는 병진 운동뿐 아니라 자세, 각속도, 짐벌, 공력, 추력 하한/상한, glide slope, pointing 제약까지 함께 들어가므로 본질적으로 비선형이고 비볼록이다. 여기에 자유 종단시간까지 포함되면 문제의 민감도가 더 커지고, **이산화 방식과 초기 궤적(initial guess)**이 수렴 여부를 크게 좌우한다.

논문은 기존 방법을 두 갈래로 정리한다. 첫째, Pontryagin Maximum Principle에 기반한 indirect method나 analytical guidance는 계산이 빠르지만, 적용 가능한 제약과 목적함수가 제한적이고 inequality constraint를 다루기 어렵다. 둘째, direct method는 SCP, SQP, SOCP 같은 형태로 일반 비볼록 문제를 다룰 수 있다. 특히 3-DoF PDG에서는 lossless convexification(LCvx)이 강력하지만, 6-DoF로 일반화하면 동역학 근사나 free-final-time 처리 때문에 정확도와 일반성이 떨어진다. 반면 SCvx는 훨씬 일반적인 비볼록 optimal control을 풀 수 있지만, 매 반복마다 convex subproblem을 풀어야 하고 초기화 품질에 크게 의존한다.

이 논문의 핵심 문제의식은 여기서 나온다. **최적해를 만드는 데 실제로 필요한 제약은 전체 제약의 일부에 불과하다.** 만약 현재 문제 파라미터와 SCvx iteration 정보만으로 “이번 subproblem에서 실제로 tight할 제약”을 미리 예측할 수 있다면, 그 제약만 포함한 작은 문제를 먼저 풀고 이를 전체 문제의 warm-start로 사용할 수 있다. 저자들은 바로 이 아이디어를 transformer로 구현하고, 기존 3-DoF T-PDG를 6-DoF SCvx로 확장한 **T-SCvx(Transformer-based Successive Convexification)**를 제안한다.

#### A. Contributions

이 논문의 기여는 세 가지로 정리할 수 있다.

1. **6-DoF Mars powered landing 문제에 T-SCvx를 적용**하여, SCvx 대비 평균 solve time을 크게 줄이면서도 full problem feasibility를 유지했다.
2. **tight constraint prediction + full solution prediction**을 동시에 사용했다. 즉, 어떤 제약이 활성화될지뿐 아니라, 상태/제어/종단시간 전체 초기 추정값까지 예측해 SCvx 수렴성을 높였다.
3. **회전 대칭 기반 data augmentation**을 도입해, 실제로 최적화를 돌려 생성한 원본 샘플 수를 2,000개 이하로 유지하면서도 6-DoF 문제에 필요한 일반화를 확보했다.

정리하면, 이 논문의 contribution은 “신경망이 최적화를 대체했다”가 아니라, **최적화의 구조를 학습해 더 작고 더 잘 초기화된 최적화 문제를 만들어 준다**는 데 있다.

#### B. Outline

논문의 흐름은 자연스럽다. 먼저 일반 비볼록 최적화와 6-DoF PDG 수식을 정리하고, SCvx가 어떤 식으로 문제를 반복적으로 convexify하는지 설명한다. 그 다음 tight constraint prediction이 왜 수학적으로 타당한지 KKT 관점에서 정리한 뒤, 이를 transformer와 결합한 T-SCvx 알고리즘을 제시한다. 마지막으로 Mars landing 셋업에서 학습/추론/성능을 검증하고, lookup table 및 KD-tree 방식과 비교해 장단점을 분석한다.

---

### 3. Methods

#### A. Non-Convex Optimization

논문은 먼저 일반적인 비볼록 최적화 문제를 다음과 같이 둔다.

$$
\min_z f(z)
$$

subject to

$$
g_i(z)=0,\quad \forall i\in I_{\mathrm{eq}}
$$

$$
g_i(z)\le 0,\quad \forall i\in I_{\mathrm{ineq}}
$$

$$
h_j(z)\le 0,\quad \forall j\in J_{\mathrm{ineq}}
$$

여기서 $z$는 discretization 이후의 유한차원 decision vector다. $g_i$는 비볼록 equality/inequality를, $h_j$는 convex inequality를 나타낸다. 이 표기는 뒤에서 6-DoF PDG를 설명할 때 매우 중요하다. 실제 착륙 문제는 “로켓 동역학”처럼 비선형인 부분과 “추력/경사/각도 제한”처럼 제약에 해당하는 부분이 한꺼번에 얽혀 있기 때문이다.

이 일반식의 역할은 두 가지다. 첫째, SCvx가 반복적으로 풀 subproblem이 무엇을 근사하는지 분명히 해 준다. 둘째, T-SCvx가 예측하는 tight constraint가 정확히 어떤 객체인지를 정의한다. 즉, 이 논문에서 예측 대상은 단순한 class label이 아니라 **최적해에서 equality와 함께 실제로 문제의 국소 구조를 결정하는 active inequality의 패턴**이다.

#### B. Problem Formulation: Six-Degree-of-Freedom Powered Descent Guidance

이 논문에서 다루는 6-DoF PDG는 다음과 같은 가정을 둔다.

* 속도가 충분히 낮아 행성 자전과 중력장 변화는 무시한다.
* 비행체는 rigid body이고 질량중심, 관성모멘트, 압력중심은 일정하다.
* 추진계는 하나의 throttleable rocket engine이며 두 축에 대해 대칭 짐벌이 가능하다.
* 공력은 일정한 대기 밀도와 ambient pressure를 이용한 모델로 포함한다.

이 설정에서 최소연료 착륙 문제의 목적함수는 다음과 같다.

$$
\min_{t_f,T_B(t)} -m(t_f)
$$

즉, 종단 질량을 최대화하는 것과 동일하므로 곧 **연료 사용량 최소화**다.

초기/종단 경계조건은 다음 의미를 갖는다.

$$
t_f\in[0,t_{f,\max}]
$$

$$
m(t_0)=m_0,\quad r_I(t_0)=r_0,\quad v_I(t_0)=v_0,\quad \omega_B(t_0)=\omega_0
$$

$$
r_I(t_f)=r_f,\quad v_I(t_f)=v_f,\quad q_{B\leftarrow I}(t_f)=q_f,\quad \omega_B(t_f)=\omega_f
$$

즉, 시작 상태는 주어지고, 끝에서는 **목표 착륙 위치/속도/자세/각속도**를 만족해야 한다.

동역학은 다음과 같이 구성된다.

$$
\dot m(t)=-\alpha_{\dot m}\|T_B(t)\|_2-\beta_{\dot m}
$$

$$
\dot r_I(t)=v_I(t)
$$

$$
\dot v_I(t)=\frac{1}{m(t)}C_{I\leftarrow B}(t)\big(T_B(t)+A_B(t)\big)+g_I
$$

$$
\dot q_{B\leftarrow I}(t)=\frac{1}{2}\Omega_{\omega_B}(t)q_{B\leftarrow I}(t)
$$

$$
J_B\dot\omega_B(t)=r_{T,B}\times T_B(t)+r_{cp,B}\times A_B(t)-\omega_B(t)\times J_B\omega_B(t)
$$

이 다섯 식은 각각 **질량 감소, 병진 위치, 병진 속도, quaternion 자세, 회전 동역학**을 의미한다. 특히 6-DoF의 어려움은 마지막 두 식에서 드러난다. 추력은 단순히 비행체를 앞으로 보내는 입력이 아니라, 짐벌과 thrust offset을 통해 자세와 각속도도 동시에 바꾸기 때문이다.

상태 제약은 다음 물리적 의미를 가진다.

$$
m_{\mathrm{dry}}\le m(t)
$$

건조질량보다 적게 연료를 소모할 수는 없다.

$$
\tan \gamma_{\mathrm{gs}}\|H_\gamma r_I(t)\|_2\le e_1\cdot r_I(t)
$$

이는 glide slope constraint로, 착륙 접근 경로가 허용된 cone 안에 머물도록 만든다.

$$
\cos\theta_{\max}\le 1-2\|H_\theta q_{B\leftarrow I}(t)\|_2^2
$$

이는 기체의 pointing angle 또는 tilt를 제한한다. 즉, 착륙 중 자세가 너무 크게 기울지 않도록 보장한다.

$$
\|\omega_B(t)\|_2\le \omega_{\max}
$$

이는 회전 속도 제한이다.

제어 제약은 다음과 같다.

$$
0<T_{\min}\le \|T_B(t)\|_2\le T_{\max}
$$

즉, 엔진은 꺼진 상태를 허용하지 않고, 최소/최대 추력 범위 안에서만 작동한다.

$$
\cos\delta_{\max}\|T_B(t)\|_2\le e_3\cdot T_B(t)
$$

이는 짐벌 또는 추력 pointing angle 제한이다. 추력 벡터가 body 축에서 너무 벗어나지 못하게 한다.

이 문제를 한 문장으로 요약하면, **자세와 병진이 강하게 결합된 자유 종단시간 최소연료 착륙 문제를, 다양한 상태/제어 제약 아래에서 푸는 것**이다. 난이도가 높은 이유는 명확하다. 추력 하한, quaternion 기반 자세, 짐벌, 공력, 종단시간이 동시에 들어가면서 문제가 강하게 비볼록해진다.

### Successive Convexification

SCvx는 위의 비볼록 optimal control problem을 한 번에 풀지 않고, 현재 iterate 주변에서 선형화한 **일련의 convex subproblem**으로 바꾸어 푼다. $k$번째 반복에서 state/control increment를

$$
d_i:=x_i-x_i^k,\qquad w_i:=u_i-u_i^k
$$

로 두면, convex subproblem은 개략적으로 다음 형태가 된다.

$$
\min_{d,w} L_k(d,w)
$$

subject to

$$
u^k+w\in U,\qquad x^k+d\in X,\qquad \|w\|\le r_k
$$

여기서 $r_k$는 trust region radius다. 즉, “현재 선형화가 믿을 만한 범위” 안에서만 해를 업데이트한다.

SCvx에서 중요한 장치는 두 가지다.

첫째, **virtual control** $v$다. 선형화된 동역학만으로는 원래 도달 가능했던 상태가 인공적으로 infeasible해질 수 있는데, $v$를 추가해 이 인공 infeasibility를 막는다.

둘째, **virtual buffer zone** $s_i'$다. 선형화된 비볼록 상태/제어 제약 역시 원래 feasible region을 잘못 잘라낼 수 있으므로, buffer를 두어 일시적으로 완화한 뒤 penalty로 다시 밀어 넣는다.

이를 포함한 1차 근사는 다음과 같다.

$$
x_{i+1}^k+d_{i+1}=f(x_i^k,u_i^k)+A_i^k d_i + B_i^k w_i + E_i^k v_i
$$

$$
s(x_i^k,u_i^k)+S_i^k d_i + Q_i^k w_i - s_i' \le 0
$$

즉, 원래 비선형 동역학과 비볼록 제약을 현재 궤적 주변의 affine model로 바꿔 푸는 것이다.

업데이트의 수용 여부는 실제 penalty cost 감소량과 선형화가 예측한 감소량의 비율로 판단한다.

$$
\Delta J_k = J(x^k,u^k)-J(x^k+d,u^k+w)
$$

$$
\Delta L_k = J(x^k,u^k)-L_k(d,w)
$$

$$
\rho_k=\frac{\Delta J_k}{\Delta L_k}
$$

* $\rho_k$가 너무 작으면 선형화가 실제 비선형 비용 감소를 과대평가한 것이므로 step을 거절하고 trust region을 줄인다.
* $\rho_k$가 적절하면 step을 수용한다.
* $\rho_k$가 크면 선형화가 잘 맞았다는 뜻이므로 trust region을 유지하거나 키운다.

논문은 SCvx의 수렴 조건도 짚는다. LICQ가 성립하면 약한 의미의 전역 수렴을, Lipschitz gradient와 Kurdyka-Lojasiewicz(KL) 성질까지 추가되면 single limit point로의 강한 수렴을 논할 수 있다. 또한 strict complementarity와 충분한 binding constraint 수가 확보되면 superlinear convergence도 기대할 수 있다. 다만 6-DoF 문제는 3-DoF bang-bang 문제와 달리 자세/공력 때문에 active set 구조가 더 복잡하므로, 실제 계산에서는 **어떤 제약이 binding할지 미리 아는 것 자체가 큰 정보**가 된다.

#### C. Tight Constraint Prediction

이 논문의 핵심은 매개변수 벡터 $\theta$와 tight constraint set 사이의 사상을 학습하는 것이다.

$$
\theta \in \Theta \subseteq \mathbb{R}^{n_p}
\quad\longmapsto\quad
\tau(\theta)\in\{0,1\}^M
$$

여기서 $\tau(\theta)$의 각 원소는 특정 inequality constraint가 최적해에서 **tight(active)**인지 아닌지를 나타내는 binary indicator다. 1이면 해당 제약이 경계에 정확히 걸리고, 0이면 slack이 있는 inactive 제약이다.

이 아이디어가 중요한 이유는, 비퇴화(non-degenerate) 조건 아래에서는 active constraints가 사실상 **support constraints**로 작동하기 때문이다. 즉, 최적해를 결정하는 국소 기하 구조는 equality와 active inequality만으로 정해진다. 이를 더 수학적으로 보면, KKT 조건에서 라그랑지안

$$
L(x,\lambda)=f(x)-\sum_{i\in I\cup E}\lambda_i c_i(x)
$$

에 대해 다음이 성립해야 한다.

1. stationarity: $\nabla_x L(x^\ast,\lambda^\ast)=0$
2. equality feasibility: $c_i(x^\ast)=0,\; i\in E$
3. inequality feasibility: $c_i(x^\ast)\ge 0,\; i\in I$
4. dual feasibility: $\lambda_i^\ast\ge 0,\; i\in I$
5. complementarity: $\lambda_i^\ast c_i(x^\ast)=0,\; i\in I\cup E$

따라서 active set은

$$
A(x)=E\cup\{i\in I \mid c_i(x)=0\}
$$

로 정의된다. 논문은 이 active set만 남긴 reduced problem이 원래 최적해를 회복한다고 주장한다. 직관은 간단하다. **inactive constraint는 이미 여유(slack)가 있으므로 최적점 근방의 feasible cone을 결정하지 못한다.** 반대로 active constraint는 조금만 움직여도 즉시 위반될 수 있으므로 최적점의 국소 구조를 결정한다. 따라서 equality와 active inequality만 포함한 reduced problem은 원래 문제와 같은 최적해를 갖는다.

논문이 제시한 증명은 Taylor 전개와 KKT를 사용한 local descent direction의 부재로 정리된다. 만약 reduced problem을 풀었을 때 원래 문제보다 더 좋은 feasible descent direction이 존재한다면, 이는 결국 원래 문제의 optimality와 모순된다. 즉, active set을 정확히 알면 전체 inequality를 다 넣지 않아도 같은 해를 얻을 수 있다. 이 수학적 사실이 곧 T-SCvx의 계산 절감 근거다.

#### D. Transformer-based Successive Convexification

기존 T-PDG는 3-DoF LCvx 문제에서 tight constraint를 예측해 convex problem을 더 빠르게 푸는 방식이었다. 이 논문은 그 아이디어를 6-DoF SCvx로 확장한다. 차이는 매우 중요하다.

* 3-DoF LCvx는 최종적으로 하나의 convex problem을 푸는 구조에 가깝다.
* 6-DoF SCvx는 **iteration마다 다른 linearization과 다른 active set**을 갖는 연쇄적 구조다.

따라서 T-SCvx는 단순히 “한 번 active set을 맞추는 모델”이 아니다. 매 iteration마다 현재 문제 파라미터와 iteration number를 보고, **이번 convex subproblem에서 어떤 제약이 tight해질지**를 다시 예측해야 한다.

논문이 제안한 T-SCvx의 핵심 흐름은 다음과 같다.

1. solution prediction network가 상태/제어/종단시간 초기 추정값을 준다.
2. SCvx의 $(k+1)$번째 subproblem에 들어가기 직전, constraint prediction network가 $\tau=f_2(\theta,k+1)$를 예측한다.
3. 예측된 tight constraint만 남겨 reduced convex subproblem을 푼다.
4. reduced solve 결과를 full problem의 warm-start로 사용하고, 최종 평가는 full penalty cost로 한다.

즉, T-SCvx는 **최적화를 대체하지 않는다.** 대신 “어떤 제약이 핵심인지”와 “어디서부터 시작해야 하는지”를 알려 줌으로써 SCvx가 훨씬 작은 문제를 더 좋은 초기조건으로 풀게 만든다.

논문은 여기에 한 가지를 더 추가한다. iteration 사이에서 예측된 tight constraint 수가 크게 바뀌면, 이는 단순히 비선형성이 큰 것만이 아니라 **문제의 활성 구조 자체가 변했다**는 뜻일 수 있다. 이를 반영하기 위해 constraint pattern 변화율

$$
\tau_r=\frac{\sum |\tau_{k+1}-\tau_k|}{\mathrm{len}(\tau_{k+1})}
$$

을 계산해 trust region contraction/growth에도 반영한다. 즉, active set이 많이 바뀔수록 다음 step을 더 보수적으로 받아들인다. 이는 T-SCvx가 단순 warm-start보다 한 단계 더 나아가, **SCvx의 trust region 설계 자체를 구조 정보로 보정**한다는 뜻이다.

### Transformer Neural Network Architecture

T-SCvx는 두 개의 transformer neural network를 사용한다.

첫째, **tight constraint prediction NN**은 문제 파라미터를 받아 어떤 inequality가 active할지 예측한다. 6-DoF PDG에 대해 입력은 17차원이며, 구체적으로 초기 속도, 초기 위치, 초기 quaternion, 초기 각속도, 초기 질량, pitch angle, glideslope angle, 그리고 SCvx iteration number를 포함한다. 출력은 $12N$ 차원의 binary vector다.

둘째, **solution prediction NN**은 초기 guess 전체를 예측한다. 입력은 iteration number를 제외한 16차원이고, 출력은 full discretized solution 전체에 해당하는 $(x,u,t_f)$다. 상태 차원이 14, 제어 차원이 3이므로 출력 크기는 대략 $17N+1$ 형태가 된다.

구조는 전형적인 transformer encoder 기반이다. 입력을 선형 encoder로 embedding space에 올리고, learned positional encoding을 더한 뒤, multi-head attention으로 변수 간 상호작용을 학습한다. 각 head의 attention은 다음과 같이 계산된다.

$$
O_h^\top
=
\mathrm{Attention}(Q_h,K_h,V_h)
=
\mathrm{Softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_k}}\right)V_h
$$

이 식의 의미는 명확하다. 입력 파라미터의 어떤 조합이 특정 constraint activation이나 trajectory shape와 강하게 연관되는지를 **가중합 형태의 attention**으로 학습하는 것이다. 6-DoF 착륙에서는 위치, 속도, 자세, 질량, 제약 각도들이 서로 얽혀 있기 때문에, 단순 MLP보다 이런 관계 모델링이 유리하다.

논문 구현은 PyTorch의 `torch.nn`을 사용했다. 중요한 점은 이 입력들이 모두 “환경/임무/상태의 파라미터”라는 것이다. 행성 상수나 기체 설계 파라미터는 고정하고, 실제 운용 중 바뀔 수 있는 초기 상태와 제약 각도만 파라미터로 삼았다. 이 설계는 T-SCvx가 **한 미션 설계 안에서 다양한 착륙 초기조건에 대한 onboard guidance generator**로 동작하도록 하기 위한 것이다.

#### E. T-SCvx Algorithm

논문은 실시간 적용 관점에서 T-SCvx를 아주 간단한 파이프라인으로 정리한다.

1. 신경망이 현재 문제 파라미터 $\theta$와 iteration 정보로 strategy를 예측한다.
2. 예측된 strategy, 즉 tight constraint set을 사용해 reduced problem을 구성한다.
3. reduced solve 결과를 SCvx 업데이트에 사용한다.
4. full problem penalty cost 기준으로 수렴과 feasibility를 판정한다.

이 구조의 장점은 안전하다. 설령 constraint prediction이 완벽하지 않더라도, T-SCvx는 **full problem을 기준으로만 최종 수렴을 인정**한다. 따라서 예측기는 최적화의 탐색을 빠르게 하는 보조장치이지, 안전 제약을 무시하는 shortcut이 아니다.

실험 데이터 생성을 위해 저자들은 SCP Toolbox의 SCvx 구현과 ECOS solver를 사용했고, 6-DoF PDG 자체는 Julia로 커스텀 구현했다. 또한 변수 스케일링을 통해 수치 조건을 개선했고, 무작정 uniform sampling을 하는 대신 feasible region을 더 효율적으로 덮는 sampling strategy를 설계했다.

### Symbolic Implementation of the 6-DoF Problem

SCvx를 적용하려면 매 iteration, 매 time step에서 동역학과 제약을 선형화해야 한다. 이를 위해 상태와 제어를 다음처럼 둔다.

$$
x(t)=
\begin{bmatrix}
r(t) & v(t) & q(t) & \omega(t) & m(t)
\end{bmatrix}^\top,
\qquad
u(t)=T(t)
$$

그리고 Jacobian을

$$
A(t)=\frac{\partial f}{\partial x},
\qquad
B(t)=\frac{\partial f}{\partial u}
$$

로 계산한다.

문제는 6-DoF에서 quaternion, skew-symmetric matrix, torque term이 함께 들어가므로 이 미분이 꽤 복잡하다는 점이다. finite difference로도 할 수는 있지만 수치 오차가 누적되기 쉽다. 그래서 논문은 Julia의 `Symbolics` 패키지를 사용해 partial derivative를 **기호적으로 정확하게** 계산한다. 이 선택은 단순 구현 취향의 문제가 아니라, SCvx처럼 반복적 선형화가 알고리즘 성능을 좌우하는 경우에 매우 실용적이다.

### Data Sampling Strategy

6-DoF 문제는 비볼록성이 강해서, 3-DoF T-PDG에서 사용하던 순진한 uniform sampling을 그대로 쓰면 데이터셋 생성 비용이 지나치게 커진다. 그래서 논문은 시스템의 회전 대칭성을 이용한다.

구체적으로는 초기 샘플을 East/North가 모두 양수인 영역에서만 뽑고, 그 다음 Up 축을 기준으로

$$
0^\circ,\ 45^\circ,\ 90^\circ,\ 135^\circ,\ 180^\circ,\ 225^\circ,\ 270^\circ,\ 315^\circ
$$

만큼 회전시킨 샘플을 합성한다. 이때 glide slope나 pointing 관련 제약은 Up 축 회전에 대해 대칭적이므로, 적절히 회전한 초기조건은 본질적으로 동일한 물리 문제를 나타낸다. 따라서 tight constraint pattern도 그 대칭성을 반영한다.

이 전략의 의미는 크다. 실제로 최적화를 돌려 얻은 원본 샘플은 1,600개 미만이지만, 회전 augmentation 후에는 11,600개가 넘는 데이터셋을 얻는다. 다시 말해, 논문은 더 좋은 신경망 구조 이전에 **문제의 물리적 symmetry를 데이터 효율로 바꾸는 방법**을 제시한다.

### Training, Validation, and Testing

학습 절차는 다음과 같다.

* 원본 + 회전 증강 데이터를 합쳐 80%는 training/validation, 20%는 test로 분리한다.
* 각 변수는 평균을 빼고 표준편차로 나누는 standardization을 적용한다.
* $K=3$인 K-fold training을 수행한다.
* warmup step은 4000, 전체 epoch은 2로 설정한다.
* solution prediction NN은 MSE loss, tight constraint prediction NN은 binary loss를 사용한다.

흥미로운 점은, 데이터셋 규모가 아주 크지 않음에도 두 네트워크가 빠르게 수렴했다는 것이다. 이는 두 가지를 시사한다. 첫째, 회전 augmentation이 실제로 정보량을 크게 늘려 주었다. 둘째, 6-DoF PDG의 active-set과 trajectory structure가 완전히 무질서한 것이 아니라, **초기 상태와 제약 파라미터의 함수로서 학습 가능한 패턴**을 가진다는 뜻이다.

---

### 4. Transformer-based Successive Convexification

#### A. Problem Setup and Parameters

실험은 Mars landing을 염두에 둔 6-DoF PDG 셋업으로 이루어진다. 모든 값은 solver conditioning을 위해 길이 $U_L$, 시간 $U_T$, 질량 $U_M$의 무차원 단위계로 정규화되었다.

핵심 설정만 정리하면 다음과 같다.

| 항목 | 값 |
|---|---|
| 중력 | $g_I=-e_1$ |
| 공력 밀도 | $\rho=0.020$ |
| 관성모멘트 | $J_B=0.01\operatorname{diag}[0.1,1,1]$ |
| 추력 범위 | $T_{\min}=0.3$, $T_{\max}=5.0$ |
| 최대 짐벌각 | $\delta_{\max}=20^\circ$ |
| 최대 각속도 | $\omega_{\max}=90^\circ/U_T$ |
| 건조질량 | $m_{\mathrm{dry}}=2.0$ |
| 이산화 노드 수 | $N=50$ |
| 최대 SCvx 반복 수 | 20 |
| discretization | FOH |
| penalty weight | $\lambda=500$ |
| solver | ECOS |
| full problem trust region init | $\eta_{\mathrm{full,init}}=2.0$ |
| reduced problem trust region init | $\eta_{\mathrm{reduced,init}}=0.01$ |

여기서 reduced problem의 초기 trust region을 full problem보다 훨씬 작게 잡은 이유가 중요하다. T-SCvx는 이미 신경망이 좋은 initial guess와 tight constraint set을 준다고 가정하므로, 초기에 크게 움직이기보다 **보수적으로 미세 조정하는 편이 안전**하기 때문이다.

데이터셋은 원본 SCvx 샘플 1,592개로 시작해, 회전 augmentation 후 11,634개가 된다. 샘플링한 파라미터는 glideslope angle, pitch angle, 초기 위치/속도, 초기 quaternion, 초기 각속도, 초기 질량이며, constraint NN에는 여기에 iteration number $k$까지 추가한다. solution NN은 16차원 입력, constraint NN은 17차원 입력을 사용한다.

네트워크 크기는 다음과 같다.

| 모델 | 입력 크기 | 출력 크기 | 주요 구조 |
|---|---:|---:|---|
| Constraint NN | 17 | $12N$ | 384-dim, 2 heads, 4 layers, dropout 0.1 |
| Solution NN | 16 | $17N+1$ | 768-dim, 2 heads, 4 layers, dropout 0.1 |

이 설정을 보면 저자들이 의도한 바가 분명하다. constraint prediction은 다중 이진 분류에 가깝기 때문에 출력 패턴을 맞추는 능력이 중요하고, solution prediction은 전체 trajectory를 회귀해야 하므로 더 큰 모델이 필요하다.

#### B. Results and Analysis

먼저 학습 성능은 다음과 같다.

| Model | Train | Validation | Test | \# of Params |
|---|---:|---:|---:|---:|
| Constraint NN | 0.024 (MSE) | 0.024 (MSE) | 96.45% (Binary Acc.) | 8.91M |
| Solution NN | 0.975 (MSE) | 1.092 (MSE) | 1.040 (MSE) | 9.0M |
| Predict Only Zeros | - | - | 95.95% (Binary Acc.) | - |
| Predict Only Ones | - | - | 4.05% (Binary Acc.) | - |

여기서 눈여겨볼 점은 constraint prediction의 binary accuracy가 96.45%라는 사실 자체보다, **all-zero baseline이 95.95%**라는 점이다. 이는 active constraint가 전체에서 희소하다는 뜻이다. 즉, 단순 정확도만 보면 큰 차이가 아닌 것처럼 보일 수 있다. 그러나 이 문제에서는 “몇 개 안 되는 truly active constraint를 얼마나 잘 찾느냐”가 계산량과 수렴성을 좌우하므로, 최종 평가는 정확도 하나가 아니라 **solver runtime과 full problem feasibility**까지 함께 봐야 한다.

실제로 545개 test sample에서 SCvx와 T-SCvx를 비교한 결과는 매우 인상적이다.

| Method | Mean Solve Time | Median Solve Time | Std. Dev. |
|---|---:|---:|---:|
| T-SCvx | 4.98 s | 4.36 s | 5.24 s |
| SCvx | 14.61 s | 14.48 s | 3.81 s |

이 결과는 다음을 의미한다.

* 평균 solve time은 약 **66% 감소**했다.
* 중앙값은 약 **70% 감소**했다.
* 절대 시간으로는 평균 약 **9.63초 단축**이다.

표준편차는 T-SCvx가 약간 더 크다. 이는 예측된 active set이 항상 완벽하게 맞지는 않기 때문에, 일부 샘플에서는 reduced solve 이후 full problem이 더 많은 교정을 요구하기 때문으로 해석할 수 있다. 그럼에도 불구하고 평균 성능 차이가 매우 크기 때문에, overall latency 측면에서는 T-SCvx의 이득이 분명하다.

논문은 세 개의 대표 trajectory도 제시한다. 이 예시들의 의미는 단순 시각화가 아니다. 서로 다른 초기 위치, 속도, 자세를 가진 6-DoF 상태에서 T-SCvx가 적절한 추력 벡터 시퀀스를 계산해 최종 착륙 상태로 유도한다는 것을 보여 준다. 즉, 모델이 특정 nominal case만 빠르게 푸는 것이 아니라, **여러 초기조건에 대해 locally fuel-optimal한 guidance trajectory를 산출할 수 있음**을 시사한다.

무엇보다 중요한 안전장치는, T-SCvx가 reduced problem만 풀고 끝나지 않는다는 점이다. 논문은 full problem penalty cost를 그대로 유지하고, 실제 penalty cost 감소가 0이 되어야만 수렴을 인정한다. 따라서 T-SCvx의 속도 향상은 “제약을 덜 검사해서 빨라진 것”이 아니라, **좋은 active-set과 좋은 initial guess 덕분에 같은 제약 검사를 더 빨리 통과한 것**이다.

#### C. Benchmarking with Table Lookup Approaches

논문은 신경망 기반 방법의 실용성을 보기 위해 lookup table 계열과도 비교한다. 비교 대상은 두 가지다.

1. **Linear interpolation-based lookup table**
2. **KD-tree nearest neighbor lookup**

공정한 비교를 위해 동일한 training/test data를 사용했지만, linear interpolation은 메모리와 추론 시간이 너무 커서 PCA로 10차원까지 줄이고, train 100개 / test 10개만 사용한 축소 조건에서 평가했다. 이 점만 봐도 6-DoF 문제에서 단순 테이블 방식이 얼마나 빨리 무거워지는지 알 수 있다.

tight constraint prediction 결과는 다음과 같다.

| Metric | Linear Interpolation | KDTree | T-SCvx |
|---|---:|---:|---:|
| Inference Time (ms) | 894.02 | 0.1500 | 4.5929 |
| MSE | 0.0368 | 0.0074 | 0.0241 |
| OOD MSE | 0.0700 | 0.0657 | 0.0373 |
| Peak Memory Usage (MB) | 1647.0 | 3.300 | 2.634 |

이 표를 읽는 핵심은 다음과 같다.

* T-SCvx는 linear interpolation 대비 추론 시간을 **99% 이상**, 메모리 사용량을 **99.8% 이상** 줄인다.
* KDTree는 in-distribution에서는 압도적으로 빠르고 정확하다.
* 그러나 OOD MSE에서는 T-SCvx가 가장 낫다.

즉, KDTree는 “가까운 샘플이 training set 안에 있을 때” 매우 강하고, T-SCvx는 **분포 밖으로 조금 벗어난 경우에도 비교적 부드럽게 일반화**한다.

solution prediction 결과도 비슷한 패턴이다.

| Metric | Linear Interpolation | KDTree | T-SCvx |
|---|---:|---:|---:|
| Inference Time (ms) | 904.43 | 0.1411 | 4.9492 |
| MSE | 0.8176 | 0.6326 | 1.0450 |
| OOD MSE | 2.3195 | 2.4128 | 1.1093 |
| Peak Memory Usage (MB) | 1678.5 | 1.5987 | 9.0118 |

여기서도 KDTree는 in-distribution MSE와 추론 속도에서 매우 강력하다. 반면 OOD MSE는 T-SCvx가 훨씬 안정적이다. 이것이 의미하는 바는 분명하다. lookup method는 **데이터베이스 안의 근처 사례를 가져오는 방식**이라 sample density가 충분하면 매우 효율적이지만, 미지의 초기조건으로 조금만 벗어나면 오차가 급증할 수 있다. 반대로 transformer는 sample 간 관계를 함수 형태로 학습하므로 extrapolation은 아니더라도 더 나은 interpolation generalization을 보여 준다.

논문은 이 결과를 RAD750 flight computer의 제약과도 비교한다. 기준은 대략 **1초 이내 runtime**과 **60 MB SRAM**이다. 이 관점에서 보면 T-SCvx와 KDTree는 둘 다 onboard 적용 가능성 범위 안에 들어오지만, linear interpolation은 메모리/시간 면에서 사실상 어렵다. 따라서 현실적인 선택지는 “OOD robustness가 중요한가, 아니면 training support 안에서의 극단적 inference speed가 중요한가”의 문제로 좁혀진다. 논문의 결론은 6-DoF PDG처럼 dispersions가 큰 문제에서는 **T-SCvx의 일반화 이점이 더 실용적**이라는 쪽에 가깝다.

---

### 5. Future Work

논문이 제시하는 미래 연구 방향은 단순히 “더 큰 모델을 쓰자”가 아니다. 오히려 T-SCvx가 제공하는 **구조 정보(active-set information)**를 최적화 알고리즘 안으로 더 깊게 집어넣는 방향이다.

첫째, trust region 설계를 더 발전시킬 수 있다. 현재도 active constraint pattern 변화율을 trust region 업데이트에 반영했지만, 앞으로는 이 정보를 더 체계적으로 이용해 nonlinearity와 active-set switching을 동시에 고려하는 state-dependent trust region으로 확장할 수 있다.

둘째, active-set 기반 nonlinear programming solver와의 결합 가능성이 있다. tight constraint prediction은 결국 “어떤 제약이 basis를 형성하는가”를 미리 맞추는 문제와 가깝기 때문에, SQP나 active-set NLP solver에 넣으면 추가적인 속도 향상을 기대할 수 있다.

셋째, 선형대수 연산의 sparsity 활용이다. 실제 대규모 trajectory optimization에서는 matrix factorization과 cone solve 비용이 큰 비중을 차지한다. 만약 tight constraint prediction이 어떤 행/열이 실제로 중요할지를 미리 알려 줄 수 있다면, custom solver 수준에서 메모리와 연산량을 동시에 줄일 수 있다.

마지막으로, 가장 실용적인 다음 단계는 flight-grade radiation-hardened hardware에서의 검증이다. 지상용 개발환경에서의 5초와 실제 우주비행 컴퓨터에서의 5초는 전혀 다른 의미를 가진다. 따라서 custom solver 구현, 병렬화 가능한 부분의 분리, 신경망 추론 스택의 경량화까지 포함한 end-to-end 검증이 필요하다.

---

### 6. Conclusion

이 논문은 6-DoF powered descent guidance를 위한 **Transformer-based Successive Convexification(T-SCvx)**를 제안하고, Mars landing 문제에서 그 효과를 실험적으로 입증한다. 핵심은 두 가지다.

첫째, transformer가 **tight constraint set**을 예측해 매 SCvx iteration에서 더 작은 reduced convex subproblem을 만들 수 있게 한다. 이는 단순 차원 축소가 아니라, KKT와 active-set 이론에 근거해 “실제로 해를 결정하는 제약만 먼저 푸는” 방식이다.

둘째, 별도의 solution prediction network가 **상태, 제어, 종단시간 전체 initial guess**를 제공해, 6-DoF SCvx에서 가장 민감한 초기화 문제를 완화한다. 따라서 T-SCvx는 문제 크기와 초기조건 품질을 동시에 개선한다.

실험적으로는 다음이 확인되었다.

* 원본 최적화 샘플 1,592개와 회전 augmentation을 통해 11,634개 데이터셋을 구성했다.
* constraint prediction은 test에서 96.45% binary accuracy를, solution prediction은 1.040 MSE를 기록했다.
* 50-node 6-DoF PDG 문제에서 T-SCvx는 SCvx 대비 평균 solve time을 약 66%, 중앙값을 약 70% 줄였다.
* linear interpolation lookup 대비 추론 시간과 메모리 사용량을 99% 이상 줄였고, OOD 상황에서는 KDTree보다 더 안정적인 오차를 보였다.
* 최종 feasibility 판정은 여전히 full problem 기준으로 수행하므로, 속도 향상이 곧 제약 완화나 안전성 저하를 의미하지 않는다.

결국 이 논문의 가장 중요한 메시지는 명확하다. **6-DoF trajectory optimization을 빠르게 만드는 가장 좋은 방법 중 하나는, 최적화 문제의 구조 자체를 학습하는 것**이다. T-SCvx는 그 구조를 active set과 initial trajectory의 형태로 추출해, 비볼록 최적화를 실시간 onboard guidance에 더 가깝게 가져온다.

---

### Acknowledgments

이 연구는 NASA Space Technology Graduate Research Opportunity와 NASA/JPL 내부 연구개발 지원을 받았다. 즉, 본 논문은 순수한 알고리즘 제안에 그치지 않고, 실제 우주탐사 적용을 염두에 둔 기관 차원의 문제의식 위에서 수행된 작업이다.

### References

References는 SCvx 이론, lossless convexification 기반 PDG, transformer trajectory optimization, lookup-table guidance, Mahalanobis distance 기반 OOD 측정, RAD750 flight computer 제약 등 본 논문의 배경을 이루는 핵심 문헌들을 연결한다. 특히 본 논문은 **convex optimization 계열의 착륙 guidance 연구**와 **학습 기반 warm-start/active-set prediction 연구** 사이를 잇는 위치에 있다.

---

**Review by 변정우, Aerospace Engineering Undergraduate Researcher**
**[Update - Time Log]**
* 2026.03.28: [Draft] 파트 1-4 리딩 완료
* 2026.0.: [ver_1] 
* 2026.0.: [Final_ver] 
