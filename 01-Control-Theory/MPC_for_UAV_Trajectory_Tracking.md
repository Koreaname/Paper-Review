# UAV 경로 추적을 위한 MPC 제어 전략 및 ROS 구현 기술 분석  
  
ETH Zurich의 Autonomous Systems Lab(ASL)에서 발표한 논문인  
**Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System**에 대한 리뷰  
  
---  
  
### 1. 논문 정보 (Reference)  
  
* **Title:** Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System  
* **Authors:** Mina Kamel, Thomas Stastny, Kostas Alexis, and Roland Siegwart  
* **Official Link:** [IEEE Xplore - MPC for UAVs using ROS](https://ieeexplore.ieee.org/document/7487277)  
  
---  
  
### 2. 핵심 제어 이론: Receding Horizon Control (RHC)  
  
논문의 핵심은 비선형 시스템의 동역학을 고려한 RHC의 실시간 최적화이다. 기본적인 상태 방정식은 다음과 같이 정의된다.  
  
$$\dot{x} = f(x, u)$$  
  
여기서 $x \in \mathbb{R}^n$은 상태 벡터, $u \in \mathbb{R}^m$은 입력 벡터이다. MPC는 매 샘플링 시점마다 유한한 Prediction Horizon $N$에 대해 다음과 같은 비용 함수를 최소화하는 최적 제어 시퀀스를 계산한다.  
  
  
  
$$\min_{U} \left( F(x_{t+N}) + \sum_{k=0}^{N-1} \|x_{t+k} - r_t\|_l + \|u_{t+k}\|_l \right)$$  
  
단, 시스템은 다음의 제약 조건을 만족해야 한다:  
* $x_{t+k+1} = f(x_{t+k}, u_{t+k})$ (시스템 동역학)  
* $u_{t+k} \in \mathbb{U_C}$, $x_{t+k} \in \mathbb{X}_C$ (입력 및 상태 제약)  
---  
  
### 3. 제어 기법의 다변화: Linear, Nonlinear, Robust  
  
논문은 UAV의 운용 목적에 따라 세 가지 범주의 MPC 전략을 심도 있게 비교 분석한다.  
  
#### 3.1. Linear MPC & Disturbance Observer  
모델링 오차 및 Disturbance를 보상하기 위해 상태 벡터에 외란 항 $d_k$를 증강하여 정밀도를 높였다.  
$$x_{k+1} = Ax_k + Bu_k + B_d d_k$$  
Observer를 통해 추정된 외란 값은 정상 상태 오차(Offset)를 제거하는 데 기여한다.  
  
#### 3.2. Nonlinear MPC (NMPC)  
UAV의 공격적인 거동(Aggressive maneuvers)시 발생하는 비선형성을 처리하기 위해 ACADO 툴킷을 사용하여 Direct Multiple Shooting 기법으로 최적화 문제를 해결한다.  
  
#### 3.3. Robust Linear MPC  
외란이 Bounded($w \in \mathbb{W}$)되어 있다고 가정하고, 최악의 경우를 고려하는 Minimax 최적화를 수행한다. 이는 특히 슬링 로드 운송과 같이 불확실성이 큰 환경에서 강력한 안정성을 제공한다.  
  
---  
  
### 4. ROS 기반의 실전 구현 (Implementation)  
  
이론적 설계를 실제 비행 기체에 이식하기 위한 하드웨어 아키텍처와 소프트웨어 통합 방안은 본 논문의 가장 실무적인 가치를 보여준다.  
  
  
  
* **계층적 제어 (Cascade Structure):** 저수준(Low-level) 자세 제어는 Pixhawk 등 MCU에서 수행하고, 고수준(High-level) 경로 추적은 NUC나 ODROID 같은 온보드 컴퓨터의 ROS 노드에서 수행하여 연산 부하를 분산한다.  
* **연산 효율화:** CVXGEN을 이용한 고속 Linear QP 솔버 생성 및 qpOASES를 활용한 실시간 최적화 해결 과정을 상세히 기술하였다.  
* **통신 지연 최소화:** ROS의 tcpNoDelay 플래그 사용 등 실제 시스템 구축 시 발생할 수 있는 Latency 문제를 해결하기 위한 구체적인 팁을 제공한다.  
  
---  
  
### 5. 결론 및 고찰
  
본 논문은 MPC가 단순히 이론적 유희를 넘어, ROS라는 범용 프레임워크 내에서 실제 드론 시스템의 성능을 어디까지 끌어올릴 수 있는지를 명확히 보여준 지침서와 같다. 특히 제어 이론의 관점에서 볼 때, 단순한 오차 추종을 넘어 제약 조건을 고려한 미래 예측이 어떻게 기체의 물리적 한계 내에서 최적의 거동을 이끌어내는지 입증한 점이 인상적이다.  
  
논문을 읽고 활용 방안을 여러 생각해본 결과로 다음과 같이 생각해볼 수 있다.  
  
**첫째, 적응형 제어와의 결합 필요성이다.** 논문의 Robust MPC는 외란의 범위를 사전에 정의해야 한다는 한계가 있다. 실제 비행 중 예상치 못한 환경 변화(돌풍 등)에 대응하기 위해, 강화학습이나 적응형 제어를 접목하여 MPC의 가중치 $Q, R$ 혹은 모델 파라미터를 실시간으로 업데이트하는 연구가 수반되어야 한다.  
  
**둘째, 임베디드 최적화의 가속화이다.** 논문에서 사용된 ACADO나 CVXGEN은 훌륭한 도구이나, 더욱 소형화된 기체(ex: Crazyflie)나 복잡한 임무 환경에서는 연산 자원이 극도로 제한된다. 따라서 최신 최적화 알고리즘(ex: OSQP 등)을 활용하여 더 긴 Prediction Horizon을 확보하면서도 실시간성을 유지하는 구현 기술이 중요하다.  
  
**셋째, 비선형 모델링의 정교화이다.** 고정익 UAV의 경우 양력 및 항력 모델링의 정확도가 제어 성능에 직결된다. 논문의 단순화된 모델을 넘어, 공력 계수를 온라인으로 식별하여 MPC 예측 모델에 실시간 반영하는 구조는 연구 및 구현하는 과정에서 중요하게 쓰인다.
  
결론적으로, 본 논문의 성과는 드론 제어 시스템의 표준 모델을 제시했다는 데 큰 의의가 있으며, 이를 바탕으로 한 지능형 자율 비행 알고리즘 설계는 향후 연구의 핵심 요소로 다뤄서 사용할 수 있다.  
  
---  
**Review by 변정우**, Aerospace Engineering Undergraduate Researcher  
  
**[Update - Time Log]**  
2026.01.24: [Draft] 전체적인 내용 초안 작성  
2026.01.: [Ver_1] Draft 리뷰에 대한 내용 보강  
2026.01.: [Final Ver] 
