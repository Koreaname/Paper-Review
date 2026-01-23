## UAV 경로 추적을 위한 MPC 제어 전략 및 ROS 구현 기술 분석  
  
### 1. 논문 정보 (Reference)  
* **Title:** Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System  
* **Authors:** Mina Kamel, Thomas Stastny, Kostas Alexis, and Roland Siegwart  
* **Official Link:** [IEEE Xplore - MPC for UAVs using ROS](https://ieeexplore.ieee.org/document/7748310)  
  
---  
  
### 2. 핵심 제어 이론: Receding Horizon Control (RHC)  
논문의 핵심은 비선형 시스템의 동역학을 고려한 RHC의 실시간 최적화이다. 기본적인 상태 방정식은 다음과 같이 정의된다.  
  
$$\dot{x} = f(x, u)$$  
  
여기서 $x \in \mathbb{R}^n$은 상태 벡터, $u \in \mathbb{R}^m$은 입력 벡터이다. MPC는 매 샘플링 시점마다 유한한 Prediction Horizon $N$에 대해 다음과 같은 비용 함수 $J_0$를 최소화하는 최적 제어 시퀀스를 계산한다.  
  
$$J_{0}(x_{0},U,X_{ref},U_{ref})=\sum_{k=0}^{N-1}(x_{k}-x_{ref,k})^{T}Q_{x}(x_{k}-x_{ref,k}) + (u_{k}-u_{ref,k})^{T}R_{u}(u_{k}-u_{ref,k})$$  
$$+ (u_{k}-u_{k-1})^{T}R_{\Delta}(u_{k}-u_{k-1}) + (x_{N}-x_{ref,N})^{T}P(x_{N}-x_{ref,N})$$  
  
**제약 조건:** * $x_{k+1}=Ax_{k}+Bu_{k}+B_{d}d_{k}$ (이산화된 시스템 동역학 및 외란 모델)  
* $u_{k} \in \mathbb{U}, x_{k} \in \mathbb{X}$ (입력 및 상태 제약 조건)  
  
폐루프 안정성을 보장하기 위해 터미널 비용 $P$와 터미널 제약 $\mathbb{X}_N$을 적절히 선택한다.  
  
---  
  
### 3. 제어 기법의 다변화: Linear, Nonlinear, Robust  
논문은 UAV의 운용 목적과 기체 특성(Multi-rotor, Fixed-wing)에 따라 세 가지 MPC 전략을 제시한다.  
  
#### 3.1. Linear MPC & Disturbance Observer  
모델링 오차 및 정상 상태 편차(Steady-state offset)를 제거하기 위해 외란 항 $d_k$를 포함한 증강 모델을 사용한다. Luenberger 형태의 Observer를 통해 외란을 실시간으로 추정한다.  
  
$$\begin{bmatrix} \hat{x}_{k+1} \\ \hat{d}_{k+1} \end{bmatrix} = \begin{bmatrix} A & B_{d} \\ 0 & I \end{bmatrix} \begin{bmatrix} \hat{x}_{k} \\ \hat{d}_{k} \end{bmatrix} + \begin{bmatrix} B \\ 0 \end{bmatrix} u_{k} + \begin{bmatrix} L_{x} \\ L_{d} \end{bmatrix} (C\hat{x}_{k} - y_{m,k})$$  
  
추정된 $\hat{d}$를 기반으로 Offset-free 타겟 상태 $x_{ref}$와 입력 $u_{ref}$를 계산하여 비용 함수에 반영한다.  
  
#### 3.2. Nonlinear MPC (NMPC)  
UAV의 고속 비행이나 급격한 거동 시 발생하는 비선형성을 직접 처리한다.  
* **연산 기법:** Direct Multiple Shooting 기술을 적용하여 연속 시간 동역학을 이산화한다.  
* **솔버:** Sequential Quadratic Programming(SQP) 방식을 채택하며, 내부 QP 문제는 qpOASES 솔버를 통해 실시간으로 해결한다.  
* **효율성:** ACADO 툴킷을 사용하여 특정 문제 구조에 최적화된 고속 C-code를 생성하여 활용한다.  
  
#### 3.3. Robust Linear MPC (RMPC)  
외란 신호가 유계($\|w\|_{\infty} \le 1$)되어 있다고 가정하는 Minimax 최적화를 수행한다.  
* **Feedback Predictions:** 보수적인 오픈루프 제어를 지양하기 위해 $U = LW + V$ 형태의 피드백 파라미터화를 도입한다.  
* **효율성:** Multiparametric-explicit 솔루션을 통해 제어 법칙을 오프라인에서 미리 계산된 Piecewise Affine 함수로 변환함으로써 연산 부하를 최소화한다.  
  
---  
  
### 4. ROS 기반의 실전 구현 (Implementation)  
이론적 알고리즘을 실시간 시스템에 통합하기 위한 구체적인 아키텍처를 제시한다.  
  
* **계층적 제어 (Cascade Structure):** Inner-loop는 Pixhawk에서 저수준 태스크를 수행하고, Outer-loop는 온보드 컴퓨터(NUC, ODROID)에서 MPC 알고리즘을 실행한다.  
* **연산 효율화:** Linear MPC는 CVXGEN을 통해 생성된 고속 솔버를 사용하며, 100Hz 수준의 고주파수 제어를 달성한다.  
* **통신 아키텍처:** MAVROS를 통해 Pixhawk와 온보드 컴퓨터 간 MAVLink 메시지를 송수신하며, 센서 퓨전 결과인 Odometry 메시지를 활용한다.  
  
---  
  
### 5. 결론 및 고찰  
본 논문은 MPC가 ROS 프레임워크 내에서 실제 드론 시스템의 성능을 극대화할 수 있음을 증명하였다. 학부연구생으로서 분석한 심화 활용 방안은 다음과 같다.  
  
첫째, **적응형 제어와의 결합** 필요성이다. RMPC의 보수적 한계를 극복하기 위해 강화학습이나 적응형 제어 기법을 결합하여 가중치 행렬($Q, R$)을 실시간 업데이트하는 연구가 수반되어야 한다.  
  
둘째, **임베디드 최적화 솔버의 고도화**이다. ACADO나 CVXGEN은 강력하나, 소형 기체에서는 OSQP와 같이 메모리 점유율이 낮고 연산 속도가 빠른 최신 솔버를 적용하여 Prediction Horizon을 더 길게 확보하는 기술이 중요하다.  
  
셋째, **비선형 모델링 및 실시간 식별의 정교화**이다. 비행 중 공력 계수를 실시간으로 식별(Online System ID)하여 MPC 모델에 반영하는 구조는 경로 추적의 정밀도를 획기적으로 향상시킬 수 있는 핵심 요소이다.  
  
**Review by 변정우, Aerospace Engineering Undergraduate Researcher** ---  
**[Update - Time Log]**  
* 2026.01.24: [Draft] 전체적인 내용 초안 작성  
* 2026.01.24: [Ver_1] 논문 상세 수식 및 하드웨어 구현 내용 심화 보강  
* 2026.01.: [Final Ver] 
