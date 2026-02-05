## UAV 경로 추적을 위한 MPC 제어 전략 및 ROS 구현 기술 분석  
  
### 0. 논문 정보 (Reference)  
* **Title:** Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System  
* **Authors:** Mina Kamel, Thomas Stastny, Kostas Alexis, and Roland Siegwart  
* **Official Link:** [IEEE Xplore - MPC for UAVs using ROS](https://ieeexplore.ieee.org/document/7748310)  
* **Open Source Code Link:** [Rotary-wing Controller](https://github.com/ethz-asl/mav_control_rw), [Fixed-wing Controller](https://github.com/ethz-asl/mav_control_fw)  
---  

### 1. Introduction  
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
\begin{bmatrix} 
\dot{\mathbf{p}}(t) \\ 
\dot{\mathbf{v}}(t) \\ 
\dot{\phi}(t) \\ 
\dot{\theta}(t) 
\end{bmatrix} = 
\begin{bmatrix} 
\mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times1} & \mathbf{0}_{3\times1} \\ 
\mathbf{0}_{3\times3} & -\text{diag}(A_x, A_y, A_z) & \begin{bmatrix} 0 \\ -g \\ 0 \end{bmatrix} & \begin{bmatrix} g \\ 0 \\ 0 \end{bmatrix} \\ 
\mathbf{0}_{1\times3} & \mathbf{0}_{1\times3} & -1/\tau_\phi & 0 \\ 
\mathbf{0}_{1\times3} & \mathbf{0}_{1\times3} & 0 & -1/\tau_\theta 
\end{bmatrix} 
\begin{bmatrix} 
\mathbf{p}(t) \\ 
\mathbf{v}(t) \\ 
\phi(t) \\ 
\theta(t) 
\end{bmatrix} + 
\begin{bmatrix} 
\mathbf{0}_{3\times1} & \mathbf{0}_{3\times1} & \mathbf{0}_{3\times1} \\ 
\mathbf{0}_{3\times1} & \mathbf{0}_{3\times1} & \mathbf{0}_{3\times1} \\ 
K_\phi/\tau_\phi & 0 & 0 \\ 
0 & K_\theta/\tau_\theta & 0 
\end{bmatrix} 
\begin{bmatrix} 
\phi_d(t) \\ 
\theta_d(t) \\ 
T(t) 
\end{bmatrix} + 
\begin{bmatrix} 
\mathbf{0}_{3\times3} \\ 
\mathbf{I}_{3\times3} \\ 
\mathbf{0}_{1\times3} \\ 
\mathbf{0}_{1\times3} 
\end{bmatrix} 
\mathbf{d}(t)
$$

로 근사화하여 yaw 동역학을 분리하고, 관성 좌표계 기반의 제어 명령을 회전 행렬 식  

$$ \begin{pmatrix}
\phi_d \\
\theta_d
\end{pmatrix} = \begin{pmatrix}
\cos \psi & \sin \psi \\
-\sin \psi & \cos \psi
\end{pmatrix} \begin{pmatrix}
{}^W \phi_d \\
{}^W \theta_d
\end{pmatrix} $$

을 통해 동체 좌표계로 변환한다.  
또한, 추적 성능 향상을 위해 목표 경로의 가속도와 기체 자세를 고려한 feed-forward 보상 식  
$\tilde{T} = \frac{T+g}{\cos \phi \cos \theta} + {}^B\ddot{z}_d, \quad \tilde{\phi}_d = \frac{g\phi_d - {}^B\ddot{y}_d}{\tilde{T}}, \quad \tilde{\theta}_d = \frac{g\theta_d + {}^B\ddot{x}_d}{\tilde{T}}$을 제어 입력에 적용한다.  
결과적으론 연속 시간 모델을 샘플링 시간 $T_s$에 따라 이산화하고, 대수 리카티 방정식을 통해 MPC의 최종 비용 행렬 $P$를 산출한다. 

## 3.2.3. ROS Integration  
제어기와 추정기를 C++ 공유 라이브러리 형태로 구현하여 ROS 노드와 인터페이싱함으로써 시스템을 통합한다. 또한, nav_msgs/Odometry 메시지로 기체 상태를 수신하고 RollPitchYawRateThrust 커스텀 메시지를 통해 제어 명령을 출력한다. 이때 단일 지점 명령 대신 전체 경로 정보를 수신함으로써, 미래 참조치를 반영하여 반응하는 MPC의 예측 제어 장점을 극대화한다. tcpNoDelay 설정을 통해 통신 지연을 최소화하며 RViz를 이용하여 목표 경로와 예측된 기체 상태를 실시간으로 시각화한다.

## 3.2.4 Experimental Results  
NUC i7 온보드 컴퓨터를 탑재한 Firefly 헥사콥터를 사용하고, Vicon 모션 캡처 시스템과 온보드 IMU 데이터를 MSF 프레임워크로 융합하여, 제어기의 실시간 성능을 실험적으로 검증한다. MPC 제어기는 100 Hz의 고주파수로 구동되며, 예측을 위한 호라이즌은 20단계로 구성한다. 이때 공격적인 경로 추적 테스트를 통해 제어 알고리즘의 안정성과 실제 비행 환경에서의 적용 가능성을 입증한다.  

### 3.3. Nonlinear MPC
비선형 비행체 모델을 통해 continuous time 비선형 모델 예측 제어기 설계하고자 한다. 이때 일반적인 최적 제어 문제(OCP)를 위해 맞춤형 C 코드 비선형 솔버를 생성성하기 위해 ACADO를 활용한다. 이에 관한 최적화 문제 수식은 다음과 같다.  

$$ \min\limits_{U,X} \int_{t=0}^{T} \left( (x(t)-x_{ref}(t))^T Q_x (x(t)-x_{ref}(t)) + (u(t)-u_{ref}(t))^T R_u (u(t)-u_{ref}(t)) \right) dt + (x(T)-x_{ref}(T))^T P (x(T)-x_{ref}(T)) $$

($$\dot{x} = f(x, u)$$, $$u(t) \in \mathbb{U}$$, $$x(0) = x(t_0)$$)  

## 3.3.1. ROS Integration  
Linear MPC와 동일한 가이드라인(3.2.3)

## 3.3.2. Experimental Results  
Linear MPC와 동일한 ROS Framework(3.2.4)

## 3.3.3. Robust Linear Model Predictive Control for Multirotor System  
Robust MPC(RMPC)를 다루기 위해 AscTec Hummingbird 쿼드로터(ASLquad)를 사용한다. 또한, 앞서 다룬 소프트웨어와 다르게, RMPC의 구현을 위해 ROS를 기반으로 하는 소프트웨어 프레임워크가 개발되었다. MATLAB은 RMPC의 explicit formulation를 유도하고 계산하는 데 사용되었으며, 알고리즘은 SIMULINK 블록 내에 구현되었다. 그 후 자동 코드 생성 기술을 통해 C-코드를 추출하여 ROS 노드에 통합한다. 실험 방식은 외부 모션 캡처를 통해 전체 상태 피드백이 제공되며 온보드 자세 및 헤딩 추정 시스템의 상대적 방향을 고려하기 위한 정렬 단계도 수행된다. 실험 세팅은 80W 전기 팬을 ASLquad로 향하게 하였고, RMPC는 난류 바람 방해에도 불구하고 나선형 경로를 추적하도록 작동하였다.  
결과적으로 추적 응답은 정밀하게 유지되며 외부 방해 요소로부터의 영향은 미미하게 관찰된다. box-constraints 설정은 제어기가 바람 방해의 동역학을 모르는 상태에서도 정밀한 나선형 경로 추적이 가능함을 보여준다.(방해되는 영향 자체를 bounded하게 처리하므로)

---  
  
### 4. Model-based Trajectory Tracking Controller for Fixed-wing
고정익 무인기의 평면상 횡방향 위치 제어를 위한 모델링, closed-loop 저수준 시스템 식별 방법론, 일반적인 형태의 high-level 비선형 모델 예측 경로 추적 제어기에 필요한 제어 목표 설계를 다룬다.

### 4.1 Fixed-wing flight dynamics & identification  
앞서 다룬 방식과 동일하게 관성 좌표계 $I$와 동체 좌표계 $B$에서 다루며, 모델에 slip이 적절하게 조절되도록 저수준 제어가 설계되었다. 이에 따라 UAV의 위치($n, e$), 요 각도($\psi$), 롤 각도($\phi$) 등을 포함한 동역학 방정식을 다음 식과 같이 다룬다.  
 $$\dot{n} = V \cos\psi + w_n$$, $$\dot{e} = V \sin\psi + w_e$$, $$\dot{\psi} = \frac{g \tan\phi}{V}$$, $$\dot{\phi} = p$$, $$\dot{p} = b_0 \phi_r - a_1 p - a_0 \phi$$  
이때 참조치($\phi_r$)에 대한 응답을 모델링하기 위해  연산 효율성을 고려하여 2차 시스템을 채택한다.(4.1.1) 이에 관하여, 차수가 증가할 때마다 제어 최적화 문제의 차원 역시 증가하므로 계산 비용이 커지게 된다.

## 4.1.1 Model identificaiton
Pixhawk와 같은 상용 오토파일럿의 폐루프 응답을 식별하기 위한 기본 방법론을 다룬다. Fig 15와 같은 PID 구조는 TECS와 NMPC에 대한 저수준 closed loop 시스템 일반적인 구조를 보여준다. 결국 자세 명령에 반응하는 동적 응답을 식별해야 하는데, 잘 튜닝된 저수준 제어기가 마련될 경우 이를 피칭 동역학 및 대기 속도 식별에도 동일한 절차로 처리할 수 있다. 이 과정에서 효율적인 데이터 수집을 위해 주파수 스윕보다 비행 시간이 짧은 2-1-1 modified doublet(동일한 비행 시간 내 더 많은 데이터 수집)을 활용하며, 주파수 스윕과 대등하게 주파수 및 식별 방식에 적합한 식별 입력을 제공한다.(데이터 세트는 논문 속 별첨된 MATLAB script를 활용)  

### 4.2 Nonlinear MPC  
ACADO 툴킷을 사용하여 최적 제어 문제를 정식화하고 고속 C 코드 기반 비선형 솔버를 생성한다. 이때 경로 자체를 이산적으로 정의한 경우, 상태 벡터는 호라이즌 내에서 스위치 상태이다. 즉, 스위치 조건이 감지될 때까지 동역학이 없으며, 감지 이후엔 계산을 위한 임의의 $\alpha$ 값이 적용된다.  

$$\dot{x}_{sw} = \begin{cases} 
\alpha, & \text{스위치 조건 충족 및 } x_{sw} > \text{임계값} \\ 
0, & \text{그 외} \end{cases}$$  

$$e_t = (d - p) \times \bar{T}_d$$  

$$e_\chi = \chi_d - \chi$$  

또한, Dubins 경로는 고정익 UAV 미션에서 원하는 비행 기동을 설명하는데 유용하다. 즉, 위 과정으로부터 Dubins 경로를 기반으로 위치 오차($e_t$)와 코스 오차($e_\chi$)를 최소화하도록 목적 함수를 구성하며, time-invariant trajectory tracking 특성을 확보한다. 이 과정에서 시간 이산화에 따른 가중치 중복 적용을 피하고자 초기 모든 references는 0으로 설정해야 한다. 이후 적용시 이차 함수로 가중치를 부여하므로 초기 호라이즌은 이후 호라이즌 값보다 더 무겁게 패널티가 부여된다.

## 4.2.1 ROS Integration  
연산 부하를 처리하기 위해 ODROID-U3 싱글 보드 컴퓨터를 탑재하고, MAVROS를 통해 Pixhawk 오토파일럿과 통신한다. 고층 NMPC 루프는 10~20Hz 주기로 구동되며, UART 직렬 통신을 통해 제어 참조치를 실시간으로 전달한다.

## 4.2.2 Experimental Results  
소형 저고도 hand-launchable Techpod UAV를 사용하여 성능을 검증한다. NMPC는 ODROID-U3에서 평균 13ms의 계산 시간을 기록하며, loiter circle로 돌아올 떄까지 overshoot는 최소한으로 관찰하여 1m 미만의 위치 오차 내로 수렴함을 알 수 있다. 또한, 고층 NPMC로부터 박스 패턴 및 복잡한 Dubins 경로 추적 실험에서 위치 오차를 3m 이내로 유지하는 정밀한 제어 성능을 입증한다.

---  
  
### 5. Conclusion 
결과적으론 멀티로터 및 고정익 UAV를 위한 선형, 비선형, Robust MPC 전략의 설계와 구현 방안을 확인하였고, 실제 비행 실험을 통해 제어 알고리즘의 성능과 안정성이 검증하였다.(ETH 취리히 ASL의 GitHub 저장소를 통해 오픈 소스 ROS 패키지로 제공)  

이에 따라 여러 가지 관점에서 내용을 정리 할 수 있다.
1. **적응형 제어와의 결합** 필요성이다. RMPC의 보수적 한계를 극복하기 위해 강화학습이나 적응형 제어 기법을 결합하여 가중치 행렬($Q, R$)을 실시간 업데이트하는 연구가 수반되어야 한다.  
  
2. **임베디드 최적화 솔버의 고도화**이다. ACADO나 CVXGEN은 강력하나, 소형 기체에서는 OSQP와 같이 메모리 점유율이 낮고 연산 속도가 빠른 최신 솔버를 적용하여 Prediction Horizon을 더 길게 확보하는 기술이 중요하다.  
  
3. **비선형 모델링 및 실시간 식별의 정교화**이다. 비행 중 공력 계수를 실시간으로 식별하여 MPC 모델에 반영하는 구조는 경로 추적의 정밀도를 획기적으로 향상시킬 수 있는 핵심 요소이다.  
  
**Review by 변정우, Aerospace Engineering Undergraduate Researcher** ---  
**[Update - Time Log]**  
* 2026.01.24: [Draft] 전체적인 내용(part 1,2,3,4,5) 리딩 완료 및 초안 작성  
* 2026.01.28: [Ver_1] part 2 수식 및 관련 내용 업데이트
* 2026.02.02: [Ver_2] part 3 수식 및 관련 내용 업데이트
* 2026.02.04: [Ver_3] part 4 수식 및 관련 내용 업데이트
* 2026.02.05: [Final Ver] 결론 업데이트 및 전체적인 내용 검토
