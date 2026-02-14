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
강화학습 문제는 일반적으로 마르코프 결정 과정(MDP)으로 정식화된다.  
에이전트는 환경과 상호작용하며 현재 상태($s_t$)에서 행동($a_t$)을 취하고, 그에 따른 보상($r_t$)을 받는다. 학습의 목표는 누적 보상의 기댓값을 최대화하는 정책($\pi$)을 찾는 것이다.  

기존의 접근법인 DDPG(Deep Deterministic Policy Gradient)와 TRPO(Trust Region Policy Optimization)는 각각 한계가 존재한다.  
* **DDPG:** Off-policy 알고리즘으로 샘플 효율성은 높으나, 불안정한 학습 특성을 보여 복잡한 쿼드로터 제어에 적용하기 어렵다.  
* **TRPO:** 확률적 정책을 사용하는 On-policy 알고리즘으로 안정적이지만, 계산 비용이 높고 수렴 속도가 느리다.  

이에 따라 본 연구에서는 결정론적 정책의 효율성과 On-policy 방법의 안정성을 결합한 새로운 학습 프레임워크를 제안한다. 이는 고차원 연속 행동 공간을 가진 로봇 제어 문제에 최적화된 접근 방식이다.  

---

### 3. Method
본 논문은 Deterministic Policy Optimization을 기반으로 하며, Natural Gradient Descent을 사용한다.  

### 3.1 Network Structure
학습을 위해 가치 네트워크와 정책 네트워크 두 가지를 사용한다.  
두 네트워크 모두 상태를 입력으로 받으며, 회전을 표현하기 위해 단위 쿼터니언 대신 회전 행렬의 9개 요소를 사용한다.  
이는 쿼터니언의 이중 커버(Double cover, $q = -q$) 문제로 인한 불연속성을 방지하기 위함이다.  
결과적으로 18차원의 상태 벡터와 4차원의 행동 벡터를 가지며, 각 네트워크는 64개의 tanh 노드를 가진 2개의 은닉층으로 구성된다.  

### 3.2 Exploration Strategy
결정론적 정책은 탐험 규칙이 명확하지 않기 때문에 효과적인 탐험 전략이 필수적이다.  
본 연구에서는 궤적을 다음 세 가지 카테고리로 분류하여 탐험을 수행한다.  
1. **Initial Trajectories:** On-policy 궤적.  
2. **Junction Trajectories:** Additive 가우시안 노이즈를 포함한 Off-policy 궤적.  
3. **Branch Trajectories:** Junction 궤적 상의 상태에서 시작하는 On-policy 궤적.  

이러한 구조는 정책과 환경이 결정론적일 때 편향되지 않은 이점 및 가치 추정을 가능하게 한다.  

### 3.3 Value Function Training & Policy Optimization
가치 함수는 온-폴리시 궤적에서 얻은 몬테카를로 샘플을 사용하여 훈련되며, 제곱 오차 대신 Huber loss를 사용하여 안정성을 높인다.  
정책 최적화는 자연 경사 하강법을 사용하며, 샘플 분포와 새로운 결정론적 정책 간의 거리를 설명하기 위해 Mahalanobis metric을 분석적 척도로 사용한다.  
특히, 신경망에서 Hessian 행렬($H_{\theta\theta}$)의 역행렬을 구하는 것은 계산 비용이 매우 높으므로, SVD(Singular Value Decomposition)를 활용한 대수적 트릭을 사용하여 Pseudo-inverse를 효율적으로 계산한다.  

$$H_{\theta\theta}^{+} \approx V(\Sigma_{v}^{+})^{2}V^{T}$$  

이 방식은 Conjugate Gradient 방법과 유사한 정확도를 보이면서도, 공분산 행렬이 Ill-condition일 때 더 안정적인 해를 제공한다.  

---

### 4. Policy Optimization in Simulation
시뮬레이션은 C++로 작성된 자체 프레임워크인 RAI(Robotic Artificial Intelligence)를 사용하였으며, 빠른 디버깅과 분석이 가능하다.  

### 4.1 Robot Model & Problem Formulation
쿼드로터는 항력을 무시한 단순 Floating body model로 가정하며, 4개의 추력이 작용한다. 운동 방정식은 다음과 같다.  

$$J^{T}T = Ma + h$$  

($J$: 자코비안, $T$: 추력, $M$: 관성 행렬, $a$: 가속도, $h$: 코리올리 및 중력)  

문제는 별도의 궤적 생성 없이 웨이포인트를 추적하는 것으로 설정된다.  
학습 중에는 관성 좌표계의 원점으로 가도록 훈련하며, 실제 운용 시에는 목표 웨이포인트와의 차이를 상태로 입력한다.  

### 4.2 Stability via PD Controller
학습 초기 단계에서 무작위 탐험으로 인해 각속도가 매우 높아져 시뮬레이션이 불안정해지는 현상을 방지하기 위해, 낮은 게인을 가진 PD 제어기를 학습 정책과 함께 사용한다.  

$$\tau_{b} = k_{p}R^{T}q + k_{d}R^{T}\omega$$  

이 PD 제어기는 학습 과정을 안정화하는 용도로만 사용되며, 최종 성능을 보장하는 주체는 아니다.  

### 4.3 Cost Function
비용 함수는 위치 오차, 각속도, 선형 속도 등에 대한 페널티로 구성되며, 위치 오차에 가장 높은 가중치를 둔다.  

$$r_{t} = 4\times10^{-3}||p_{t}|| + 2\times10^{-4}||a_{t}|| + 3\times10^{-4}||\omega_{t}|| + 5\times10^{-4}||v_{t}||$$  

Discount factor $\gamma$는 0.99로 설정되었다.  

### 4.4 Training Performance
제안된 알고리즘은 DDPG 및 TRPO와 비교되었다.  
DDPG는 적절한 성능으로 수렴하지 못했고, TRPO는 제안된 방법과 유사한 성능(비용 0.2~0.25)에 도달했으나 훨씬 긴 시간이 소요되었다.  
제안된 방법은 보수적인 파라미터 설정에도 불구하고 Parallel rollouts 덕분에 반복당 10초 미만의 빠른 학습 속도를 보였다.  

---

### 5. Experiments
훈련된 정책은 추가적인 최적화 없이 실제 쿼드로터(AscTec Hummingbird)에 적용되었다.  

### 5.1 Computational Efficiency
정책 네트워크의 평가 시간은 약 $7\mu s$로 측정되었으며, 이는 선형 MPC 제어기가 약 $1,000\mu s$ 소요되는 것에 비해 압도적으로 효율적이다.  
이는 기체의 온보드 컴퓨터 자원을 다른 알고리즘(상태 추정 등)에 할애할 수 있게 한다.  

### 5.2 Tracking Performance & Recovery
웨이포인트 추적 실험에서 약간의 추적 오차가 발생했으나, 이는 정책이 실제 환경의 외란 없이 훈련되었기 때문이다.  
반면, 복구 테스트에서는 놀라운 성능을 보였다. 기체를 거꾸로 5m/s의 속도로 던지는 가혹한 초기화 상황에서도 안정적으로 자세를 회복하고 호버링에 성공했다.  
흥미롭게도 실제 환경에서의 복구 동작이 시뮬레이션보다 더 안정적이었는데, 이는 시뮬레이션에서 모델링되지 않은 공기 저항과 자이로 효과가 감쇠 역할을 했기 때문으로 분석된다.  

---

### 6. Conclusion
본 연구는 모델 프리 방식으로 훈련된 신경망 정책이 쿼드로터 제어에 있어 뛰어난 성능과 계산 효율성을 동시에 달성할 수 있음을 입증하였다.  
제안된 결정론적 온-폴리시 알고리즘은 기존의 TRPO나 DDPG보다 계산 효율성과 안정성 면에서 우수한 결과를 보였다.  

향후 연구 방향으로는 모델링 오차에 자동으로 적응할 수 있는 RNN(Recurrent Neural Network)의 도입과, 실제 시스템에서의 전이 학습을 통해 미지의 동역학적 특성을 포착하여 성능을 더욱 개선하는 것이 제시된다.  
  
**Review by 변정우, Aerospace Engineering Undergraduate Researcher ---  
[Update - Time Log]**
* 2026.02.13: [Draft] 전체적인 내용(part 1,2,3,4,5) 리딩 완료 및 초안 작성  
* 2026.02.: [ver_1]
