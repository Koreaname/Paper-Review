## UAV 경로 추적을 위한 MPC 제어 전략 및 ROS 구현 기술 분석  
  
### 0. 논문 정보 (Reference)  
* **Title:** Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System  
* **Authors:** Mina Kamel, Thomas Stastny, Kostas Alexis, and Roland Siegwart  
* **Official Link:** [IEEE Xplore - MPC for UAVs using ROS](https://ieeexplore.ieee.org/document/7748310)  
* **Open Source Code Link:** [Rotary-wing Controller](https://github.com/ethz-asl/mav_control_rw), [Fixed-wing Controller](https://github.com/ethz-asl/mav_control_fw)  
---  

### 1. 서론 및 개요 (Introduction)  
본 연구는 로봇 운영 체제(ROS)를 기반으로 무인 항공기(UAV, Unmanned Aerial Vehicle)의 정밀한 궤적 추적을 실현하기 위한 모델 예측 제어(MPC, Model Predictive Controller) 전략을 다룬다. 아래 내용은 크게 멀티로터 시스템과 고정익 UAV 두 가지 기종에 대한 MPC 설계 및 실시간 구현 방법을 기술한다. 

---  
  
### 2. 핵심 제어 이론

### 2.1 Receding Horizon Control(RHC, 후퇴 지평선 제어)  
논문의 핵심은 비선형 시스템의 동역학을 고려한 RHC의 실시간 최적화이다. RHC는 고정된 지평선 최적화의 한계를 극복하기 위해 제안된 전략이다.  
이는 전체 제어 시퀀스를 계산하되, 오직 첫 번째 단계의 제어 입력($u_t^*$)만을 시스템에 적용하고 다음 시점에 이 과정을 반복하는 방식이다.  
 
RHC의 일반적인 최적화 문제는 다음과 같이 표현할 수 있다.  

$$ \begin{aligned} \min_{Z} \quad & F(x_{t+N}) + \sum_{k=0}^{N-1} \Vert x_{t+k} - r_t \Vert_l + \Vert u_{t+k} \Vert_l \end{aligned} $$  

(제약 조건: 시스템 동역학 $x_{t+k+1} = f(x_{t+k}, u_{t+k})$, 입력 제약 $u_{t+k} \in U_C$, 상태 제약 $x_{t+k} \in X_C$  
터미널 비용 $F(x_{t+N})$: 예측 지평선의 끝에서 시스템의 안정성을 보장하기 위한 가중치)  

### 2.2 Linear MPC  
  
$$J_{0}(x_{0},U,X_{ref},U_{ref})=\sum_{k=0}^{N-1}(x_{k}-x_{ref,k})^{T}Q_{x}(x_{k}-x_{ref,k}) + (u_{k}-u_{ref,k})^{T}R_{u}(u_{k}-u_{ref,k})$$  
$$+ (u_{k}-u_{k-1})^{T}R_{\Delta}(u_{k}-u_{k-1}) + (x_{N}-x_{ref,N})^{T}P(x_{N}-x_{ref,N})$$  
  
($Q_x$: 상태 오차에 대한 가중치, $R_u$: 제어 입력 오차에 대한 가중치, $R_\Delta$: 제어 입력 변화율에 대한 가중치(좀 더 부드러운 제어를 위해 사용), $P$: 최종 상태 오차에 대한 가중치)  

이를 통해 시스템의 실제 측정값과 모델 예측값의 차이를 이용해 외부 외란 $\hat{d}$를 추정한다.  
즉, 정상 상태(Steady-state)에서 목표 궤적을 정확히 따라갈 수 있는 $x_{ref}$와 $u_{ref}$를 계산한다.  

### 2.3 Nonlinear MPC  

$$ \begin{aligned} \min_{U,X} \quad & \int_{0}^{T} \Vert h(x(t), u(t)) - y_{ref} \Vert_Q^2 dt + \Vert h(x(T)) - y_{N,ref} \Vert_{Q_N}^2 \end{aligned} $$

($Q_x$: 상태 오차(State Error)에 대한 가중치, $R_u$: 제어 입력 오차(Control Input Error)에 대한 가중치, $R_\Delta$: 제어 입력 변화율에 대한 가중치, $P$: 최종 상태 오차(Terminal State Error)에 대한 가중치)  

항공기는 일반적으로 비선형 특성을 다룬다. 이는 Direct multiple shooting 기법을 통해 비선형 계획법 문제로 변환하며, qpOASES와 같은 솔버를 활용하여 실시간으로 해결한다.

### 2.4 Linear Robust MPC
Robust MPC는 외란이 존재하더라도 최악의 상황을 고려하여 제어하는 방식이다. 이는 아무리 최악인 외란이 존재하여도 무조건 기체가 머물게 하겠다는 보수적인 비행 전략인 셈이다. 따라서 robust하게 다루기 위해 최소 피크 성능 지표인 MPPM(Minimum Peak Performance Measure)을 활용한다.  
$x_{k+1} = Ax_k + Bu_k + Gw_k$($w_k$는 범위가 제한된 외란)인 시스템 모델로부터 다음 식과 같이 예측된 상태 $\mathcal{X}$는 현재 상태, 미래 제어 입력, 외란에 선형적으로 의존한다.  
$\mathcal{X} = \mathcal{A}x_{k|k} + \mathcal{B}\mathcal{U} + \mathcal{G}\mathcal{W}$  
$\mathcal{Y} = \mathcal{C}\mathcal{X}$
 
이는 다음과 같은 방식으로 다룰 수 있다.
1. Feedback Predictions  
: 단순히 오픈 루프 제어 시퀀스를 계산하면 제어 결과가 지나치게 보수적(움직임이 너무 둔함)이 될 수 있다. 따라서 이를 해결하기 위해 피드백 예측 기법을 도입한다.
$\mathcal{X} = \mathcal{A}x_{k|k} + \mathcal{B}\mathcal{V} + (\mathcal{G}+\mathcal{B}\mathcal{L})\mathcal{W}$, $\mathcal{U} = \mathcal{L}\mathcal{W} + \mathcal{V}$으로 수식을 재구성하고, 이 매개변수화를 통해 Minimax MPC 문제를 효율적으로 풀 수 있는 컨벡스 최적화 문제로 변환할 수 있다.  

2. Robust Uncertain Inequalities Satisfaction
3. Robust State and Input Constraints
4. Minimum Peak Performance Robust MPC formulation  
2,3,4: 외란 $w$가 범위 내의 어떤 값을 갖더라도 제약 조건을 만족해야 한다. 앞서 수식을 범위로 다루는 과정에서 수식 속 절댓값 항을 처리하기 위해 행렬 변수 $\Gamma$와 $\Omega$를 도입한다. 이에 따라 입력 제약($\mathcal{F}_u$)과 상태 제약($\mathcal{F}_x$)을 외란에 상관없이 항상 만족하도록 부등식을 재구성한다. 

5. Multiparametric Explicit Solution  
: 앞선 부등식 범위에 따라 RMPC는 매 순간 선형 프로그래밍 문제를 풀어야 하므로 연산량이 많다는 단점이 있다. 그러므로 상태 공간을 여러 개의 Polyhedral 영역으로 나누면, 각 영역에서 제어 입력은 $u_k = F^r x_k + Z^r$과 같은 단순한 선형 형태가 된다. 이는 복잡한 최적화 문제를 실시간으로 푸는 대신, 드론이 현재 어느 영역에 있는지만 확인해서 미리 저장된 공식을 쓰면 된다. 즉, 연산 능력이 매우 낮은 마이크로컨트롤러에서도 초고속 실행을 가능하게 한다. 

---  
  
### 3. MPC for Multi-rotor Systems  
멀티로터 시스템의 정밀한 궤적 추종을 위한 MPC 프레임워크를 논하고자 한다. 전체 시스템은 모델링, Linear MPC, Nonlinear MPC, Robust MPC로 확장되며, 각 단계는 이론적 설계와 ROS 기반의 실시간 구현 및 검증 과정을 포함한다.

### 3.1. Multirotor System Model  
멀티로터를 6자유도(DoF)를 가진 강체로 모델링하고자 한다. 고정된 관성 좌표계 $W$와 기체에 부착된 동체 좌표계 $B$에 대하여, 좌표계 $B$의 원점의 위치를 $p$와 회전 행렬을 $R$로 나타낸다. 또한, 기체의 상태 벡터 $x$는 위치, 속도, 자세(Roll, Pitch, Yaw), 각속도로 정의되며, 제어 입력 $u$는 총 추력(T)과 자세각 명령(Roll, Pitch)으로 구성된다.  
이 모델에서는 1차 시스템 거동을 통해 원하는 롤 및 피치 각도 $\phi_d, \theta_d$를 추적할 수 있는 저수준 자세 제어기가 있다고 가정한다. 이러한 1차 inner-loop 근사는 MPC가 저수준 제어기의 거동을 고려할 수 있도록 충분한 정보를 제공한다. 이때, 내측 루프의 1차 파라미터는 전형적인 시스템 식별 기법을 통해 식별될 수 있다.  

### 3.2. Linear MPC  
Linear MPC을 공식으로 어떻게 다루냐에 따른 멀티 로터 시스템의 궤도 추적과 ROS로의 통합 과정을 다루고자 한다. 이는 다루고자 하는 최적화 문제에 대하여 CVXGEN freamework를 사용하여 C-code 솔버를 만들어서 해결할 수 있다. CVXGEN은 convex optimization 문제를 해결하기 위한 고속 솔버를 생성한다. 
$\min\limits_{U, X} \left( \sum_{k=0}^{N-1} (x_k - x_{ref,k})^T Q_x (x_k - x_{ref,k}) + (u_k - u_{ref,k})^T R_u (u_k - u_{ref,k}) + (u_k - u_{k-1})^T R_{\Delta} (u_k - u_{k-1}) \right) + (x_N - x_{ref,N})^T P (x_N - x_{ref,N})$  
해당 식과 여러 제약 조건들을 활용하여 솔버를 생성할 수 있고, 이에 관한 아이디어들은 다음과 같다.  

## 3.2.1. Attitude Loop Parameters Identification  
우선 제어 대상 시스템의 선형 모델이 필요하다. 시간에 따른 입력 고도와 실제 고도를 준비하고, 가능한 축에 대해 최대한 많은 작업을 수행한다. 이를 통한 타당성을 검증하여 모델 파라미터 식별을 위한 타당성 퍼센티지를 확인한다.

## 3.2.2. Linearization, Decoupling and Discretization  
호버링 조건 및 작은 자세각 가정 하에 비선형 시스템을 선형화된 상태 공간 방정식 식  
$$
\begin{aligned}
\begin{bmatrix} \dot{p}(t) \\ \dot{v}(t) \\ \dot{\phi}(t) \\ \dot{\theta}(t) \end{bmatrix} &= \begin{bmatrix} 0_{3\times3} & I_{3\times3} & 0_{3\times1} & 0_{3\times1} \\ 0_{3\times3} & -\text{diag}(A_{x}, A_{y}, A_{z}) & \begin{smallmatrix} 0 \\ -g \\ 0 \end{smallmatrix} & \begin{smallmatrix} g \\ 0 \\ 0 \end{smallmatrix} \\ 0_{1\times3} & 0_{1\times3} & -1/\tau_{\phi} & 0 \\ 0_{1\times3} & 0_{1\times3} & 0 & -1/\tau_{\theta} \end{bmatrix} \begin{bmatrix} p(t) \\ v(t) \\ \phi(t) \\ \theta(t) \end{bmatrix} \\
&\quad + \begin{bmatrix} 0_{3\times3} & 0_{3\times3} \\ 0_{3\times2} & 0_{3\times1} \\ K_{\phi}/\tau_{\phi} & 0 \\ 0 & K_{\theta}/\tau_{\theta} \end{bmatrix} \begin{bmatrix} \phi_{d}(t) \\ \theta_{d}(t) \\ T(t) \end{bmatrix} + \begin{bmatrix} 0_{3\times3} \\ I_{3\times3} \\ 0 \\ 0 \end{bmatrix} d(t)
\end{aligned}
$$로 근사화하기.  
yaw 동역학을 분리하고, 관성 좌표계 기반의 제어 명령을 회전 행렬 식  
$$ \begin{aligned}\begin{pmatrix} \phi_{d} \\ \theta_{d} \end{pmatrix} &= \begin{pmatrix} \cos \psi & \sin \psi \\ -\sin \psi & \cos \psi \end{pmatrix} \begin{pmatrix} {}^{W}\phi_{d} \\ {}^{W}\theta_{d} \end{pmatrix}\end{aligned} $$을 통해 기체 바디 좌표계로 변환.  
추적 성능 향상을 위해 목표 경로의 가속도와 기체 자세를 고려한 feed-forward 보상 식  
$\tilde{T} = \frac{T+g}{\cos \phi \cos \theta} + {}^B\ddot{z}_d, \quad \tilde{\phi}_d = \frac{g\phi_d - {}^B\ddot{y}_d}{\tilde{T}}, \quad \tilde{\theta}_d = \frac{g\theta_d + {}^B\ddot{x}_d}{\tilde{T}}$을 제어 입력에 적용.  
연속 시간 모델을 샘플링 시간 $T_s$에 따라 이산화하고, 대수 리카티 방정식을 통해 MPC의 최종 비용 행렬 $P$를 산출. 

## 3.2.3. ROS Integration  
제어기와 추정기를 C++ 공유 라이브러리 형태로 구현하여 ROS 노드와 인터페이싱함으로써 시스템을 통합한다. 또한, nav_msgs/Odometry 메시지로 기체 상태를 수신하고 RollPitchYawRateThrust 커스텀 메시지를 통해 제어 명령을 출력한다. 이때 단일 지점 명령 대신 전체 경로 정보를 수신함으로써, 미래 참조치를 반영하여 반응하는 MPC의 예측 제어 장점을 극대화한다. tcpNoDelay 설정을 통해 통신 지연을 최소화하며 RViz를 이용하여 목표 경로와 예측된 기체 상태를 실시간으로 시각화한다.

## 3.2.4 Experimental Results  
NUC i7 온보드 컴퓨터를 탑재한 Firefly 헥사콥터를 사용하고, Vicon 모션 캡처 시스템과 온보드 IMU 데이터를 MSF 프레임워크로 융합하여, 제어기의 실시간 성능을 실험적으로 검증한다. MPC 제어기는 100 Hz의 고주파수로 구동되며, 예측을 위한 호라이즌은 20단계(steps)로 구성한다. 이때 공격적인 경로 추적 테스트를 통해 제어 알고리즘의 안정성과 실제 비행 환경에서의 적용 가능성을 입증한다.  

### 3.3. Nonlinear MPC
선형 모델의 한계를 극복하고 기체의 비선형 동역학을 온전히 활용하기 위해 NMPC를 적용한다. 이는 고속 비행이나 급격한 기동과 같이 선형화 가정이 깨지는 영역에서도 우수한 제어 성능을 보장한다.  

## 3.3.1. ROS Integration  
Nonlinear MPC의 구현은 Linear MPC와 유사한 ROS 통신 구조를 따르지만, 핵심인 최적화 솔버에서 차이가 있다.  
- ACADO Toolkit: 실시간 비선형 최적화를 수행하기 위해 ACADO Toolkit을 사용한다. 이는 최적화 문제를 기호적으로 분석하여 고도로 최적화된 C 코드를 생성해준다.  
- Code Generation: 생성된 코드는 qpOASES와 같은 QP 솔버와 결합되어 ROS 노드 내에서 수 밀리초(ms) 이내에 비선형 최적해를 도출한다. 이를 통해 온보드 컴퓨터의 제한된 연산 자원 내에서도 Receding Horizon Control을 실시간으로 수행할 수 있다.  

## 3.3.2. Experimental Results  
NMPC의 검증은 Linear MPC보다 더욱 Aggressive maneuvers인 시나리오에서 수행되었다. 예를 들어, 고속의 8자 비행이나 Helix 상승 비행과 같은 과도한 자세 변화가 요구되는 궤적에서 실험을 진행하였다. 결과적으로 NMPC는 전체 비행 영역에서 예측 모델의 정확성을 유지하며, Linear MPC 대비 향상된 추종 정밀도와 동적 반응성을 보여주었다.  

## 3.3.3. Robust Linear Model Predictive Control for Multirotor System  
실제 환경에서는 바람과 같은 예측 불가능한 외란이 존재하므로, 이에 대한 Robustness가 요구된다. 이에 관하여 튜브 기반(Tube-based) 또는 Minimax 접근 방식을 응용한 Robust Linear MPC를 다루고자 한다.

- 실험 셋업: 구조적으로 개조된 'AscTec Hummingbird (ASLquad)'를 사용하였으며, 강력한 외란 조건을 모사하기 위해 80W급 전기 팬을 사용하여 기체에 지속적인 풍압을 가했다.

- 검증 결과: 강인 제어기는 외란이 존재하는 상황에서도 상태 변수가 허용된 튜브 내에 머물도록 제어 입력을 조절하였다. 특히 팬 앞을 지나가는 나선형 궤적 실험과 무거운 화물을 매달고 비행하는 실험에서, 일반적인 MPC 대비 월등히 안정적인 궤적 유지 성능을 확인하였다.

---  
  
### 4. Model-based Trajectory Tracking Controller for Fixed-wing
UAVs  


### 4.1 Fixed-wing flight dynamics & identification  
 

## 4.1.1 Model identificaiton


### 4.2 Nonlinear MPC  


## 4.2.1 ROS Integration  


## 4.2.2 Experimental Results  



---  
  
### 5. 결론 및 고찰  
본 논문은 MPC가 ROS 프레임워크 내에서 실제 드론 시스템의 성능을 극대화할 수 있음을 증명하였다. 학부연구생으로서 분석한 심화 활용 방안은 다음과 같다.  
  
첫째, **적응형 제어와의 결합** 필요성이다. RMPC의 보수적 한계를 극복하기 위해 강화학습이나 적응형 제어 기법을 결합하여 가중치 행렬($Q, R$)을 실시간 업데이트하는 연구가 수반되어야 한다.  
  
둘째, **임베디드 최적화 솔버의 고도화**이다. ACADO나 CVXGEN은 강력하나, 소형 기체에서는 OSQP와 같이 메모리 점유율이 낮고 연산 속도가 빠른 최신 솔버를 적용하여 Prediction Horizon을 더 길게 확보하는 기술이 중요하다.  
  
셋째, **비선형 모델링 및 실시간 식별의 정교화**이다. 비행 중 공력 계수를 실시간으로 식별(Online System ID)하여 MPC 모델에 반영하는 구조는 경로 추적의 정밀도를 획기적으로 향상시킬 수 있는 핵심 요소이다.  
  
**Review by 변정우, Aerospace Engineering Undergraduate Researcher** ---  
**[Update - Time Log]**  
* 2026.01.24: [Draft] 전체적인 내용(part 1,2,3,4,5) 리딩 완료 및 초안 작성  
* 2026.01.28: [Ver_1] part 2 수식 및 관련 내용 추가
* 2026.02.02: [Ver_2] part 3 수식 및 관련 내용 추가
* 2026.01.: [Final Ver] 
