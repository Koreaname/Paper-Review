## 강화학습 기반의 쿼드로터 제어 및 결정론적 온-폴리시 학습 알고리즘 분석  

### 0. 논문 정보 (Reference)  
* **Title:** Control of a Quadrotor with Reinforcement Learning  
* **Authors:** Jemin Hwangbo, Inkyu Sa, Roland Siegwart, and Marco Hutter  
* **Official Link:** [IEEE Xplore - Control of a Quadrotor with Reinforcement Learning](https://ieeexplore.ieee.org/document/7945231)  
* **Open Source Code Link:** [RAI Framework](https://bitbucket.org/leggedrobotics/rai)  

---

### 1. Introduction  
본 연구는 강화학습 기술을 이용하여 훈련된 신경망으로 쿼드로터를 제어하는 방법을 제시한다.  
기존의 로봇 제어 방식이 상위 레벨의 궤적 생성과 하위 레벨의 액추에이터 제어를 분리했던 것과 달리, 
논문 속 방법은 상태를 actuator command로 직접 맵핑하는 방식을 통해 사전 정의된 제어 구조 없이도 복잡한 제어가 가능함을 보여준다. 이는 쿼드로터 제어시 step response와 호버링에 대한 동적 움직임(안정성)에 성과를 보인다.

특히 새롭게 제안된 알고리즘인 Deterministic On-policy 학습 알고리즘은 zero-bias, zero-variance이므로, 복잡한 작업에 대해 보수적이지만 안정적인 학습 성능을 보이며 신경망 계산에 부담이 적어, 기존 알고리즘 대비 쿼드로터 제어에 더 적합한 특성을 가진다. 랜덤한 초기 설정으로부터 recovery를 입증하는 성과를 내었다. 또한, 정책 평가에 소요되는 연산 시간은 타임 스텝당 $7\mu s$에 불과하여, 근사 모델을 사용하는 일반적인 궤적 최적화 알고리즘보다 두 자릿수 이상 빠른 연산 속도를 달성하였다.  

---

### 2. Background
현재 활용되는 Deterministic policy 최적화는 natural gradient descent를 이용한다. 이는 확률론적 policy보다 낮은 분산(더 안정적), gradient 작성시 단순화, 예측 가능 범위(확률론적은 예측 불가능)이라는 장점으로 인해 활용된다. 이에 따라 Deterministic policy gradient method는 상태 공간을 탐색하는데에 분명한 규칙이 없으므로 좋은 탐색 전략이 필요하다.

한편, 강화학습 과정에서는 환경 모델을 사전에 알 수 없는 블랙박스 시스템이나 실제 로봇으로부터 샘플 데이터를 수집한다. 초기 상태 분포 $d_{0}(s)$부터 $T$ 스텝 동안 행동 $a \in \mathcal{A}$를 선택하면 그에 따른 궤적 $(s_{1:T+1}, a_{1:T}, r_{1:T})$을 얻을 수 있다. 여기서 $s \in \mathcal{S}$는 시스템의 상태, $r \in \mathbb{R}$은 현재 상태와 행동에 의해 결정되는 deterministic cost function $R:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$의 반환값을 의미한다.

이 알고리즘 최종 목표는 상태 분포 전반에 걸친 평균 가치를 최소화하는 parameterized policy $\pi_{\theta}$를 찾는 것이다.($\theta$는 policy 파라미터) 학습이 안정적으로 이루어지도록 discount factor $\gamma \in [0, 1)$를 적용하여 상태 가치 함수 $V$(미래의 보상 합)가 무한히 커지지 않고 유한한 값을 가지도록 제한한다. 상태 분포 내의 감가를 무시하고 식을 단순화할 경우, $\theta$에 대한 deterministic policy gradient 수식은 다음과 같이 정의할 수 있다.( $$V^{\pi_\theta}(s) = \mathop{\mathbb{E}}\limits_{s_{t+1}, s_{t+2}, \dots} \left[ \sum_{t=t}^{\infty} \gamma^t r_t \mid s_t = s, \pi_\theta \right]$$ )

$$L(\pi_\theta) = \mathop{\mathbb{E}}\limits_{s_0, s_1, \dots} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \right] = \int_{\mathcal{S}} d^{\pi_\theta}(s) V^{\pi_\theta}(s) ds$$  

$$\nabla_\theta L(\pi_\theta) = \mathop{\mathbb{E}}\limits_{s \sim d^{\pi_\theta}(s)} \left[ \nabla_\theta \pi_\theta(s) \nabla_a Q^{\pi_\theta}(s, a) \mid a = \pi_\theta(s) \right]$$

위 식에서 action-value 함수 $$Q^{\pi_{\theta}}(s_{t},a_{t}) = \mathop{\mathbb{E}}\limits_{s_{t+1}} [r(s_{t},a_{t}) + \gamma V^{\pi_{\theta}}(s_{t+1})]$$ 는 특정 상태 $s_t$에서 임의의 행동 $a_t$를 먼저 취한 후, 이후부터는 계속해서 현재의 $\pi_{\theta}$를 따랐을 때 얻게 되는 기대 가치를 의미한다. 

여기에 baseline function 역할을 하는 state value 함수 $V^{\pi_{\theta}}(s)$를 도입하면, 앞선 수식을 advantage function $A^{\pi_{\theta}}(s,a) = Q^{\pi_{\theta}}(s,a) - V^{\pi_{\theta}}(s)$를 활용한 형태로 다시 쓸 수 있다.

$$\nabla_{\theta}L(\pi_{\theta})=\mathbb{E}_{s\sim d^{\pi_{\theta}}(s)}[\nabla_{\theta}\pi_{\theta}(s)\nabla_{a}A^{\pi_{\theta}}(s,a)|a=\pi_{\theta}(s)]$$

이 식의 결과는 임의의 특정 행동 $a$를 취했을 때 얻는 가치에서 현재 $\pi_{\theta}(s)$가 기본적으로 지시하는 행동을 취했을 때 얻는 가치를 뺀 값으로, 해당 행동을 통해 얻을 수 있는 추가적인 가치적 이득을 나타낸다.

---

### 3. Method
본 섹션에서는 쿼드로터 제어 정책 훈련을 위해 고안된 알고리즘과 구체적인 방법론을 설명한다.  

### A. Network Structure  
훈련 과정에는 가치 네트워크(value network)와 정책 네트워크(policy network)라는 두 가지 신경망 구조가 사용된다.  

두 네트워크는 공통적으로 18차원의 상태(state) 벡터를 입력으로 받는다. 상태 벡터는 위치, 선형 속도, 각속도와 더불어 회전을 나타내는 9개의 요소로 구성된 회전 행렬 $R_{b}$를 포함하며, 정규 분포를 따르도록 적절히 스케일링된다. 로보틱스에서 회전 매개변수화에 단위 쿼터니언(unit quaternion)을 자주 사용하지만, 본 연구에서는 이를 배제하였다. 단위 쿼터니언은 $q = -q$와 같이 동일한 회전을 두 개의 값으로 표현하는 특징이 있어 훈련 데이터가 두 배로 요구되거나 정의역을 $S^{3}$의 반구로 제한할 때 불연속 함수가 생성되는 문제점이 존재하기 때문이다. 반면 회전 행렬은 표현의 중복성은 있으나 구조가 단순하여 이러한 함정에서 자유롭다.  

네트워크의 출력은 로터의 추력(rotor thrusts)을 나타내는 4차원의 행동(action) 벡터이다. 두 신경망 모두 64개의 tanh 노드로 구성된 2개의 은닉층(hidden layer)을 갖는 구조로 설계되었다.  

### B. Exploration Strategy  
탐색(Exploration) 전략은 결정론적 정책 기울기 최적화에서 가장 중요한 부분 중 하나이다. 본 연구에서는 궤적을 초기 궤적(initial trajectories), 교차 궤적(junction trajectories), 분기 궤적(branch trajectories)의 세 가지 범주로 나누어 탐색을 수행한다.  

* 초기 및 분기 궤적: 온폴리시(on-policy)를 따르는 궤적이다.  
* 교차 궤적: 공분산 $\Sigma$를 갖는 가산 가우스 노이즈(additive Gaussian noise)가 부가된 오프폴리시(off-policy) 궤적이다. 분기 궤적은 이 교차 궤적 상의 특정 상태에서부터 시작하여 다시 온폴리시를 따른다.  

이러한 설계의 목적은 정책과 환경이 모두 결정론적일 때, 편향되지 않은(unbiased) 이점 및 가치 추정치를 얻기 위함이다. 교차 궤적의 길이를 단일 타임스텝보다 길게 가져감으로써 신경망이 잘 근사하지 못하는 샘플링 영역의 경계 문제를 완화하고, 보다 넓게 분포된 샘플을 통해 알고리즘이 국소 최솟값(local minimum)에 빠질 확률을 낮춘다.  

### C. Value Function Training  
가치 함수는 온폴리시 궤적에서 추출한 몬테카를로(Monte-Carlo) 샘플을 기반으로 훈련된다. 궤적이 유한한 길이를 가지므로, 종단 상태의 가치는 현재 근사된 가치 함수 파라미터 $\eta$를 갖는 $V(s_{T}|\eta)$를 통해 추정된다.  

$$v_{i} = \sum_{t=i}^{T-1} \gamma^{t-i} r_{t}^{p} + \gamma^{T-i} V(s_{T}|\eta)$$  

시스템에 노이즈가 없는 완전 결정론적 환경이므로, 에피소드 종료 지점에서 업데이트를 수행할 때 위와 같은 몬테카를로 방식이 시간 차 학습(temporal difference learning)이나 $TD(\lambda)$ 방식보다 본 연구의 환경에서 우수한 결과를 제공한다. 최적화 과정에서는 오차 제곱 함수 대신 이상치에 강건한 Huber loss를 사용하며, 손실값이 0.0001 미만으로 떨어질 때까지 최대 200회의 반복(iteration) 업데이트를 수행한다.  

### D. Policy Optimization  
정책 최적화는 자연 기울기 하강법(natural gradient descent)을 활용한다. 기존 확률론적 정책에서는 샘플 분포와 정책 간의 거리를 평균 KL 발산(Kullback-Leibler divergence)으로 측정하지만, 본 연구의 결정론적 정책에서는 행동 분포와 새로운 정책 간의 분석적 거리를 의미하는 Mahalanobis 메트릭을 거리 척도로 채택한다.  

정책 최적화 과정은 다음과 같은 수식으로 정의된다.  

$$A^{\pi}(s_{i},a_{i}^{f}) = r_{i}^{f} + \gamma v_{i+1}^{f} - v_{i}^{p}$$  
$$\overline{L}(\theta) = \sum_{k=0}^{K} A^{\pi}(s_{k},\pi(s_{k}|\theta))$$  
$$\theta_{j+1} \leftarrow \theta_{j} - \frac{\alpha}{K} \sum_{k=0}^{K} n_{k}$$  
$$\text{s.t. } (\alpha n_{k})^{T}D_{\theta\theta}(\alpha n_{k}) < \delta, \quad \forall k$$  

여기서 $n_{k}$는 샘플당 자연 기울기(per-sample natural gradient)로, $D_{\theta\theta}n_{k} = g_{k}$를 만족하는 벡터이다. $D_{\theta\theta}$는 정책 파라미터 $\theta$에 대한 제곱된 Mahalanobis 거리의 Hessian 행렬이며, $g_{k}$는 샘플당 추정된 일반 기울기이다. 부등식 제약 조건은 신뢰 영역 제약(trust region constraint)으로, 작은 노이즈 벡터에 대해 기울기가 과도하게 커지는 현상을 방지하여 각 샘플의 업데이트 영향력을 제한한다.  

신경망 구조에서 Hessian 행렬 $H_{\theta\theta} = \frac{\partial a}{\partial \theta}^{T}D_{aa}\frac{\partial a}{\partial \theta} = J^{T}D_{aa}J$는 완전 계수(full rank)가 아니므로 일반적인 역행렬 연산이 불가능하다. TRPO는 켤레 기울기법(conjugate gradient)을 통해 근사해를 구하지만, 본 접근법에서는 특이값 분해(SVD)를 사용하여 다음과 같이 대수적으로 정확한 의사역행렬(pseudoinverse)을 계산한다.  

$$H_{\theta\theta}^{+} = V(\Sigma_{v}^{+})^{2}V^{T}$$  

위 수식에서 $L_{aa}$는 대칭 및 양의 정부호(positive-definite) 속성을 지닌 $D_{aa}$의 숄레스키 인자(Cholesky factor)이며, $L_{aa}^{T}J = V\Sigma_{v}U^{T}$와 같이 Thin SVD를 적용하여 연산량을 획기적으로 감축시킨다. 탐색 노이즈 공분산 $\Sigma$의 경우, 정책 기울기를 통해 최적화하지 않고 행동 공간의 스케일에 맞추어 수동으로 정의하여 탐색 성능을 안정적으로 보장한다.  

### 4. Policy optimization in simulation
본 연구에서는 시뮬레이션 및 빠른 연산을 위해 C++ 기반의 자체 소프트웨어 프레임워크인 RAI(Robotic Artificial Intelligence)를 개발하여 사용하였다. 동역학 적분 연산에 소요되는 시간은 네트워크 훈련 시간에 비해 극히 적다.  

### A. Robot model for simulation  
쿼드로터 동역학을 시뮬레이션하기 위해 동체에 작용하는 모든 항력을 무시하고 4개의 추력만 작용하는 단순한 부동체(floating body) 물리 모델을 사용하였다. 운동 방정식은 다음과 같이 정의된다.  

$$J^{T}T = Ma + h$$  

여기서 $J$는 로터 중심의 누적 야코비안(stacked Jacobian) 행렬, $T$는 추력, $M$은 관성 행렬, $a$는 일반화된 가속도, $h$는 코리올리 및 중력 효과를 의미한다. 프로펠러는 양의 힘(위쪽 방향 추력)만 생성할 수 있으므로, 시뮬레이션에서 음의 추력이 감지되면 이를 0으로 임계치(threshold) 처리한다. 또한 쿼드로터의 극단적인 동적 모션 시 발생하는 적분 부정확성을 개선하기 위해 boxplus 연산자를 사용하여 0.01초라는 큰 타임스텝에서도 정확한 동역학 적분이 가능하도록 설계하였다.  

### B. Problem Formulation  
제어의 주된 목표는 별도의 궤적 생성 없이 웨이포인트(waypoint)를 추적하는 것이다. 이를 위해 정책 최적화 중에는 관성 좌표계의 원점으로 이동하도록 훈련하며, 쿼드로터가 물리적으로 가능한 모든 상태(예: 임의의 선형 및 각속도를 지닌 채 뒤집힌 상태)에서 스스로 안정화할 수 있도록 무작위 초기 상태에서 학습을 진행한다.  

학습 과정을 안정화하기 위해 낮은 이득(gain)을 갖는 오일러 각 기반의 단순한 PD 제어기를 결합하였다. 제어기 수식은 다음과 같다.  

$$\tau_{b} = k_{p}R^{T}q + k_{d}R^{T}\omega$$  

이 PD 제어기는 쿼드로터가 뒤집혀 있을 때 신경망이 관측하는 기울기가 매우 낮고 불연속적인 문제를 완화해 주지만, 최종적이고 정교한 제어 동작은 모두 학습된 정책 네트워크가 전담한다. 훈련에 사용되는 비용 함수(cost function)는 다음과 같이 정의된다.  

$$r_{t} = 4\times10^{-3}||p_{t}|| + 2\times10^{-4}||a_{t}|| + 3\times10^{-4}||\omega_{t}|| + 5\times10^{-4}||v_{t}||$$  

위 식에서 알 수 있듯 위치 오차($p_{t}$)에 가장 높은 비용 계수를 부여하여 추적 성능을 극대화하였으며, 감가율(discount factor) $\gamma$는 0.99로 설정하였다.  

### C. Network Training  
가치 네트워크는 온폴리시 샘플을 활용하여 훈련된다. 빠른 수렴보다는 안정적이고 신뢰성 높은 수렴에 초점을 맞추어 매우 보수적인 알고리즘 파라미터(초기 궤적 512개, 분기 궤적 1024개, 노이즈 깊이 2)를 설정하였으며, 이는 반복(iteration)당 100만 타임스텝의 데이터를 생성한다.  

롤아웃 시뮬레이션을 배치 단위로 병렬화한 덕분에 이러한 방대한 연산량에도 불구하고 반복당 훈련 시간은 10초 미만으로 단축되었다. 다른 알고리즘과 성능을 비교한 결과, DDPG는 합리적인 시간 내에 적절한 성능으로 수렴하지 못했고, TRPO는 제안된 방법과 동일한 성능(비용 0.2~0.25)을 달성했으나 수렴에 훨씬 더 긴 시간이 소요되었다.  

### D. Performance in Simulation  
학습된 정책의 안정성을 검증하기 위해 $SO(3)$ 공간에서 균등하게 샘플링된 회전 상태 및 무작위 속도 등 극한의 초기 상태에서 복구(recovery) 성능을 테스트하였다. 쿼드로터가 바닥에 닿는 것을 '실패'로 규정했을 때, 선형 MPC 제어기는 100회 중 71%의 높은 실패율을 보인 반면, 학습된 신경망 정책은 단 4%의 실패율만을 기록하며 압도적인 신뢰성을 입증하였다.  

또한, Eigen 라이브러리를 활용해 구현된 정책 네트워크는 한 번의 타임스텝에서 상태를 평가하는 데 단 $7 \mu s$만이 소요되어 연산 비용이 사실상 거의 들지 않는다. 이는 1 타임스텝당 약 $1000 \mu s$가 소요되는 선형 MPC 제어기와 비교하여 실시간 로봇 제어에 있어 엄청난 연산 효율성을 제공한다.

---

### 5. Experiments
본 연구에서는 시뮬레이션에서 학습된 정책(policy)을 추가적인 최적화 과정 없이 실제 쿼드로터 비행 하드웨어에 직접 적용하여 성능을 평가하였다.  

실험에는 Ascending Technologies 사의 Hummingbird 쿼드로터가 사용되었으며, 기체의 질량은 0.665 kg, 관성 모멘트는 $I_{xx} = 0.007$, $I_{yy} = 0.007$, $I_{zz} = 0.012$ 기반으로 구성되었다. 기체에는 탑재 연산(onboard calculation)을 위해 0.059 kg의 무게를 갖는 1.44 GHz 쿼드코어 Atom 프로세서 기반의 Intel Compute Stick이 장착되었다.  

상태 정보(state information)는 Vicon 모션 캡처 시스템을 통해 획득하였으며, Vicon 시스템의 통신 지연 및 낮은 업데이트 빈도를 보완하기 위해 온보드 IMU 측정값을 다중 센서 융합(multi-sensor fusion) 프레임워크와 결합하여 사용하였다.  

시뮬레이션 모델과 실제 하드웨어 환경 간에는 다음과 같은 주요한 동역학적 차이점이 존재하였다.  

* 첫째, 모터 제어기가 조절하는 실제 모터 속도를 정확히 알 수 없으며, 로터가 가속될 때는 비교적 빠르지만 감속될 때는 천천히 반응하는 비대칭적 특성이 있었다.  
* 둘째, 지면 근처 비행 시 발생하는 공기역학적 변화(지면 효과 등)가 시뮬레이션의 단순화를 위해 배제되었다.  
* 셋째, 배터리 잔량의 감소나 배터리 교체로 인한 무게 중심 변화 등 비행 중 동적 파라미터가 지속적으로 변동하여 호버링(hovering) 로터 속도에 영향을 미쳤다.  
* 넷째, 무선 통신 지연 및 상태 추정 오차가 전체 시스템 성능에 영향을 주었다.  

이러한 차이에도 불구하고, 1 m $\times$ 1 m 크기의 사각형 꼭짓점 4개를 추적하는 웨이포인트 추적 테스트를 수행한 결과, 약간의 추적 오차(tracking error)만 발생할 뿐 안정적인 비행이 가능함을 확인하였다. 학습 시 다양한 외부 외란(external disturbances)을 학습 데이터에 포함하지 않았기 때문에, 높은 이득(high gain)을 사용하는 기존의 고전 제어기보다는 추적 오차가 다소 높게 나타났다.  

가장 극단적인 상황을 모사하기 위해 약 5 m/s의 초기 선속도로 쿼드로터를 뒤집어서 허공에 직접 던지는 수동 자세 회복 실험(manual launch test)을 진행하였다. 기체가 추락하기 시작할 때 제어기를 활성화하였으며, 예상과 달리 실제 환경의 쿼드로터가 시뮬레이션보다 오히려 더 부드럽고 안정적인 자세 제어 동작을 보여주었다. 이는 시뮬레이션 모델에 포함되지 않았던 공기 항력(air drag)과 로터의 자이로스코프 효과(gyroscopic effect)가 고속 회전 시 시스템의 모션을 물리적으로 안정화하는 데 긍정적으로 기여했기 때문으로 분석된다.  

### 6. Conclusion 
본 논문은 모델 프리(model-free) 방식으로 훈련된 신경망을 이용한 쿼드로터 제어 정책을 성공적으로 제시하였다. 시뮬레이션 환경은 기본 물리 모델을 바탕으로 구축되었으나, 실제 정책 훈련 과정에서는 해당 모델의 구조적 특성을 역으로 이용하지 않았다. 이를 통해 시스템 모델을 정교하게 분석하고 복잡한 제어기 구조를 사전 설계해야 하는 기존의 엔지니어링 부담을 크게 해소할 수 있었다. 학습된 정책은 뛰어난 비행 제어 성능을 보여주었을 뿐만 아니라 정책 평가에 소요되는 연산 비용 또한 극도로 낮아 실시간 임베디드 제어 환경에 매우 적합하다.  

특히 본 연구에서 제안한 새로운 학습 알고리즘은 결정론적 환경의 이점을 살려 신경망 훈련 단계를 최소화하는 보수적(conservative)이고 효율적인 최적화 방식으로, DDPG 및 TRPO와 같은 기존 알고리즘 대비 연산 시간과 수렴 속도 측면에서 탁월한 우위를 입증하였다. 훈련 중 발산(divergence) 문제가 발생하지 않는 높은 안정성을 지니므로 복잡한 로보틱스 제어 문제에 폭넓게 적용될 수 있다.  

시뮬레이션 환경에서의 웨이포인트 추적 테스트 결과, 상수로 쉽게 보정 가능한 1.3 cm 수준의 미세한 정상 상태 오차(steady state error)만이 관찰되었다. 또한 실제 하드웨어 수동 투척 복구 테스트에서는 공기역학적 특성(항력 및 자이로스코프 효과)의 상호작용을 통해 시뮬레이션보다 더욱 뛰어난 안정성을 확보하며 복구에 성공하였다.  

향후 연구 방향으로는 시스템 파라미터 추정 기법을 활용하여 시뮬레이션에 더욱 정교한 물리 모델을 도입하는 방안을 모색할 예정이다. 나아가, 모델링 오차에 스스로 적응하여 동적인 보정이 가능한 순환 신경망(RNN) 구조를 도입하거나, 실제 비행 시스템 데이터 기반의 전이 학습(transfer learning)을 수행하여 모델링 되지 않은 완전히 미지의 동역학적 특성까지 제어 정책에 반영하는 것을 장기적인 목표로 삼고 있다.
  
**Review by 변정우, Aerospace Engineering Undergraduate Researcher ---  
[Update - Time Log]**
* 2026.02.13: [Draft] 전체적인 내용(part 1,2,3,4,5) 리딩 완료 및 초안 작성  
* 2026.02.16: [ver_1] part 1, 2 수식 및 관련 내용 업데이트
* 2026.02.: [ver_2]
