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
쿼드로터 제어 policy 훈련을 위한 알고리즘 및 방법론을 다룬다.  

### A. Network Structure  
훈련 과정에는 value network와 policy network라는 두 가지 신경망 구조가 사용된다. 이 두 네트워크는 공통적으로 18차원의 상태 벡터를 입력으로 받는다. 이 상태 벡터는 위치, 선형 속도, 각속도와 회전을 나타내는 9개의 요소로 구성된 회전 행렬 $R_{b}$를 포함하며, 정규 분포를 따르도록 적절히 스케일링된다.  

보통 회전 매개변수화에 단위 쿼터니언을 자주 사용하지만 해당 이론에선 배제한다. 단위 쿼터니언은 $q = -q$와 같이 동일한 회전을 두 개의 값으로 표현하는 특징이 있어, 훈련 데이터가 두 배로 요구되거나 정의역을 $S^{3}$의 반구로 제한할 때 불연속 함수가 생성되는 문제점이 존재하기 때문이다. 반면 회전 행렬은 표현의 중복성은 있으나 구조가 단순하여 이러한 문제점에는 자유롭다.  

반면 네트워크의 출력은 로터의 추력을 나타내는 4차원의 행동 벡터이다. 두 신경망 모두 64개의 tanh 노드로 구성된 2개의 은닉층을 갖는 구조로 설계되었다. 이는 특정 문제에 맞추어 노드나 층의 수를 최적화하는 과정을 거치지 않았음에도, 오직 하나의 구조로 다양한 문제에 범용적으로 대처할 수 있음을 확인하였다.

### B. Exploration Strategy  
탐색 전략은 Deterministic policy 기울기 최적화에서 가장 중요한 부분 중 하나이다. 이는 궤적을 초기 궤적(initial), 교차 궤적(junction), 분기 궤적(branch)로 나누어 탐색을 수행한다.  

* 초기, 분기 궤적: on-policy를 따르는 궤적.  
* 교차 궤적: 공분산 $\Sigma$를 갖는 가산 가우스 노이즈가 부가된 off-policy 궤적. 이때 분기 궤적은 이 교차 궤적 상의 특정 상태에서부터 시작하여 다시 on-policy를 따른다.  

이 일련의 과정은 정책과 환경이 모두 결정론적일 때 편향되지 않은 이점 및 가치 추정치를 얻기 위해 설계되었다. 각 궤적의 특징과 길이 설정에 따른 효과는 다음과 같다.  

* 교차 궤적의 길이 설정
  * 효과:  길이를 단일 타임스텝보다 길게 가져감으로써 신경망이 잘 근사하지 못하는 샘플링 영역의 경계 문제를 완화하고, 보다 넓게 분포된 샘플을 통해 알고리즘이 local minimum에 빠질 확률을 낮춘다.  
  * 한계점: 교차 궤적이 너무 길어지면 해당 궤적이 상태 분포 $d^{\pi}(s)$를 따른다는 가정을 위반하게 된다. 하지만 초기 궤적보다 충분히 짧다면 실제 성능에는 악영향을 미치지 않는다.  
  * 튜닝 파라미터: 논문에선 무작위 초기화가 상태 탐색 문제를 효과적으로 해결해주었기 때문에 교차 궤적의 길이가 탐색 성능에 큰 차이를 만들지 않았다. 따라서 이는 상황에 따라 조절 가능한 튜닝 파라미터로 작용한다.  

* 분기 궤적 및 전체 궤적의 길이 설정
  * Terminal States 가치: 모든 궤적은 유한한 길이를 가지므로 terminal states의 가치 비용은 근사된 가치 함수 $V(s|\eta)$를 통해 추산된다.  
  * 긴 분기 궤적의 장단점: 긴 분기 궤적을 사용하면 학습 단계에서 더 많은 평가 연산이 요구되지만 추정치의 편향을 낮출 수 있다.  
  * 환경적 특성: 특히 본 시뮬레이션 환경은 노이즈가 없고 분산이 0이므로, 궤적이 길수록 이점 추정의 정확도가 항상 높아진다.  
  * 결론: 본 연구는 빠른 수렴보다는 안정적이고 신뢰성 높은 수렴에 초점을 맞추었기에 긴 궤적을 채택하였으며, 노이즈가 있는 시스템의 경우 그에 맞는 적당한 궤적 길이가 요구된다.

### C. Value Function Training  
가치 함수는 on-policy 궤적에서 추출한 몬테카를로 샘플을 기반으로 훈련된다.  
궤적이 유한한 길이를 가지므로, 종단 상태의 가치는 현재 근사된 가치 함수의 파라미터 $\eta$를 활용해 $V(s_{T}|\eta)$로 추산되며, 이를 수식으로 나타내면 다음과 같다.  

$$v_{i}=\sum_{t=i}^{T-1}\gamma^{t-i}r_{t}^{p}+\gamma^{T-i}V(s_{T}|\eta)$$  

($T$는 궤적의 길이)  
시스템에 노이즈가 없는 완전 결정론적 환경에서는 에피소드 종료 지점에서만 업데이트를 수행하는 이러한 몬테카를로 방식이 시간 차 학습이나 $TD(\lambda)$ 방식보다 더 좋은 결과를 제공한다. 가치 함수 최적화 과정에서는 오차 제곱 함수 대신 이상치에 강건한 Huber loss를 사용하여 학습 안정성을 높이며, 손실값이 0.0001 미만으로 떨어질 때까지 최대 200회의 반복 업데이트를 수행한다.  

### D. Policy Optimization  
  
결정론적 강화학습에서 정책 최적화는 natural gradient descent에 기반한다. 확률론적 정책에서는 평균 KL 발산을 사용하여 두 확률분포 간의 거리를 정의하지만 마할라노비스 거리를 사용하여 새로 업데이트된 결정론적 정책과 행동 분포 간의 거리를 정의한다. 이는 샘플 분포를 명시적으로 추정하지 않고도 정책 파라미터 공간의 거리 척도를 정의할 수 있게 해준다.  
이에 대한 최적화 문제는 다음과 같이 설정된다.  
  
$$A^{\pi}(s_i,a_i^f) = r_i^f + \gamma v_{i+1}^f - v_i^p$$  
$$\overline{L}(\theta) = \sum_{k=0}^{K} A^{\pi}(s_k,\pi(s_k|\theta))$$  
$$\theta_{j+1} \leftarrow \theta_j - \frac{\alpha}{K}\sum_{k=0}^{K} n_k$$  
  
($A^{\pi}$는 어드밴티지 함수, $r_i^f$는 즉시 보상, $v_i^p$는 현재 가치 함수 예측, $\gamma$는 discount factor)  
$n_k$는 샘플별 자연 기울기(per-sample natural gradient)로서, 정책 기울기 $g_k$와 Hessian $D_{\theta\theta}$ 사이의 선형 방정식 $$D_{\theta\theta} n_k = g_k$$ 를 만족한다. 또한 신뢰 영역 제약을 적용하여 업데이트의 크기를 제한한다.  
$$(\alpha n_k)^T D_{\theta\theta} (\alpha n_k) < \delta$$  
결정론적 정책에서는 Hessian이 full rank가 아니므로 일반적인 역행렬 연산이 불가능하다. 따라서 얇은 특이값 분해와 숄레스키 분해를 이용하여 pseudoinverse를 계산한다.  
$$H_{\theta\theta}^{+} = V(\Sigma_v^{+})^2 V^T$$  
이 방식은 conjugate gradient 기반 방법보다 계산 효율성이 높으며, 안정적인 업데이트를 보장한다. 탐색 노이즈 공분산 $\Sigma$는 정책 최적화의 변수가 아니라 탐색을 위한 상수로 취급하며, 행동 공간의 스케일을 고려하여 수동으로 설정한다.  

정책 최적화 절차는 다음과 같다.    
1. 정책 파라미터 $\theta$와 가치 함수 파라미터 $\eta$ 초기화  
2. 탐색 전략에 따라 on-policy 및 branch 궤적 수집  
3. 몬테카를로 추정치를 이용해 가치 함수 $V(s|\eta)$를 Huber loss로 업데이트  
4. 자연 기울기 계산 후 신뢰 영역 제약을 적용하여 정책 업데이트  
5. 수렴 시까지 반복  
  
---  
  
### 4. Policy Optimization in Simulation  
정책 학습과 검증을 위해 C++ 기반의 RAI(Robotic Artificial Intelligence) 프레임워크를 사용하였다.  
시뮬레이션과 네트워크 학습을 동일한 환경에서 수행하며, 동역학 적분 시간은 네트워크 학습 시간에 비해 매우 작도록 설계되었다.  

### 4.1 Robot Model for Simulation  
시뮬레이션 모델은 공기 저항을 무시한 단순한 부동체 모델을 사용한다. 네 개의 로터 추력만을 고려하여 운동을 계산한다.  
$$J^T T = M a + h$$  
($J$는 로터 중심의 누적 야코비안 행렬, $T$는 로터 추력 벡터, $M$은 관성 행렬, $a$는 일반화된 가속도, $h$는 코리올리 및 중력 항)  
로터는 양의 추력만 생성하므로 음의 추력은 0으로 임계치 처리한다. 또한 boxplus 연산자를 이용하여 0.01 s의 큰 타임스텝에서도 안정적인 적분이 가능하도록 하였다.  
  
### 4.2 Problem Formulation  
정책은 웨이포인트 없이 목표 위치를 추적하도록 학습된다. 훈련 시에는 관성 좌표계 원점으로 이동하도록 학습하며, 실제 운용 시에는 상태에서 목표 위치를 뺀 상대 상태를 입력한다.  
이때 훈련 안정화를 위해 저이득 PD 제어기를 보조적으로 사용한다.    
$$\tau_b = k_p R^T q + k_d R^T \omega$$  
($\tau_b$는 가상 토크, $R$은 회전 행렬, $q$는 자세 오차, $\omega$는 각속도, z축 게인은 다른 축의 1/6로 설정)  
훈련에 사용된 비용 함수는 다음과 같다.  
$$r_t = 4\times 10^{-3}\|p_t\| + 2\times 10^{-4}\|a_t\| + 3\times 10^{-4}\|\omega_t\| + 5\times 10^{-4}\|v_t\|$$  
위치 오차 항에 가장 큰 가중치를 부여하였으며, discount factor $\gamma$는 0.99로 설정하였다.  

### 4.3 Network Training    
가치 함수는 on-policy 궤적에서 추출한 몬테카를로 샘플을 기반으로 학습한다. 빠른 수렴보다는 안정적인 수렴을 목표로 하여, 초기 궤적 512개와 분기 궤적 1024개를 사용하였다.  
반복당 약 100만 타임스텝을 생성하지만 병렬화된 롤아웃 덕분에 반복당 학습 시간은 10초 미만이다.  
결과적으론 DDPG는 적절한 성능으로 수렴하지 못했고, TRPO는 유사한 성능을 달성했으나 더 많은 연산 시간을 필요로 했다. 제안된 방법은 연산 효율성과 안정성 측면에서 우수하였다.  
  
### 4.4 Performance in Simulation    
$SO(3)$ 공간에서 무작위로 샘플링한 초기 상태에서 복구 성능을 평가하였다.(이때 실패는 기체가 지면에 닿는 경우로 정의)  
선형 MPC의 실패율은 71%였으나, 학습된 정책은 4%의 낮은 실패율을 기록하였다. 정책 네트워크는 Eigen 기반 행렬 연산으로 구현되어 상태 평가에 약 7 µs가 소요되었다. 이는 선형 MPC의 약 1 ms 대비 두 자릿수 이상 빠른 속도이다.  
  
---  
  
### 5. Experiments  
학습된 정책은 추가적인 최적화 없이 실제 쿼드로터에 적용되었다. Ascending Technologies사의 Hummingbird 기체를 사용하였으며, 질량은 0.665 kg, 관성 모멘트는 $I_{xx}=0.007$, $I_{yy}=0.007$, $I_{zz}=0.012$ kg·m²이다. 상태 정보는 Vicon 모션 캡처 시스템과 온보드 IMU 데이터를 다중 센서 융합하여 획득하였다.  
또한 웨이포인트 추적 실험(1 m × 1 m 사각형 경로)에서 안정적인 비행을 확인하였다. 정상 상태 오차는 약 1.3 cm 수준이었다. 수동 자세 회복 실험에서는 약 5 m/s 초기 속도로 뒤집힌 상태에서 회복에 성공하였다. 실제 환경에서는 공기 항력과 자이로 효과로 인해 시뮬레이션보다 더 안정적인 거동을 보였다.  

---  
  
### 6. Conclusion  
해당 논문은 모델 프리 방식으로 훈련된 신경망을 통해 쿼드로터를 직접 제어하는 방법을 제시하였다. 상태를 로터 추력으로 직접 맵핑함으로써 사전 정의된 제어 구조 없이도 복잡한 기동이 가능함을 보였다. 마할라노비스 거리와 SVD 기반 pseudoinverse 계산을 통해 안정적이고 효율적인 자연 기울기 최적화를 달성하였다. 정책 평가는 수 마이크로초 수준으로, 실시간 임베디드 제어에 매우 적합하다.  

이에 관한 향후 논의 사항은 다음과 같이 정리할 수 있다.  
1. 보다 정밀한 물리 모델 도입 및 파라미터 추정  
2. 순환 신경망(RNN)을 통한 동역학 적응  
3. 실제 데이터 기반 전이 학습을 통한 성능 향상  

**Review by 변정우, Aerospace Engineering Undergraduate Researcher ---  
[Update - Time Log]**
* 2026.02.13: [Draft] 전체적인 내용(part 1,2,3,4,5) 리딩 완료 및 초안 작성  
* 2026.02.16: [ver_1] part 1, 2 수식 및 관련 내용 업데이트
* 2026.02.24: [ver_2] part 3 수식 및 관련 내용 업데이트
* 2026.02.27: [Final ver] 나머지 내용 내용 업데이트 및 전체적으로 검토
