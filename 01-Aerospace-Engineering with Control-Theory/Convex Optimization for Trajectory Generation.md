## Convex Optimization for Trajectory Generation — Dynamically Feasible Trajectory를 위한 Convex Optimization Tutorial

### 0. 논문 정보 (Reference)

* **Title:** Convex Optimization for Trajectory Generation: A Tutorial On Generating Dynamically Feasible Trajectories Reliably And Efficiently
* **Authors:** Danylo Malyuta, Taylor P. Reynolds, Michael Szmuk, Thomas Lew, Riccardo Bonalli, Marco Pavone, Behçet Açıkmeşe
* **Venue / Year:** Preprint / arXiv, 2021
* **Keywords:** Trajectory Generation, Optimal Control, Convex Optimization, Lossless Convexification, LCvx, Sequential Convex Programming, SCvx, GuSTO, Trust Region, Virtual Control, Powered Descent Guidance
* **Code / Project:** 논문 Figure 2에서 numerical examples를 재현할 수 있는 open-source implementation을 제공한다고 명시한다. 구체적으로는 `dmalyuta/scp_traj_opt` repository의 `csm` branch를 가리킨다.

---

### Abstract

이 논문은 autonomous vehicle과 robot이 실제 환경에서 안전하게 움직이기 위해 필요한 **trajectory generation** 문제를 convex optimization 관점에서 정리한 tutorial paper이다. 여기서 trajectory는 단순한 geometric path가 아니라, 시간에 따른 state trajectory $x(t)$와 control trajectory $u(t)$의 조합이다. 따라서 좋은 trajectory는 목적지만 연결하는 선이 아니라, system dynamics, state/control constraints, boundary conditions를 모두 만족하는 **dynamically feasible trajectory**여야 한다.

문제는 이러한 trajectory generation이 일반적으로 continuous-time optimal control problem으로 표현되며, 대부분 **nonconvex**라는 점이다. Nonconvex optimal control problem을 일반 NLP solver로 직접 풀면 local optimum, solver failure, infeasibility certificate 부재, runtime 불안정성 문제가 생길 수 있다. 특히 onboard autonomy에서는 계산 자원과 안전성이 모두 중요하므로, 일반적인 nonlinear optimization만으로는 충분하지 않다.

이 논문은 이 문제를 해결하기 위해 세 가지 convex optimization 기반 접근을 통합적으로 설명한다. 첫 번째는 **LCvx(lossless convexification)**이다. LCvx는 특정 구조의 nonconvex control constraint를 slack variable과 lifting을 통해 convex relaxation으로 바꾸고, 최적해에서 그 relaxation이 original problem의 feasible solution을 회복함을 보인다. 두 번째와 세 번째는 **SCvx(successive convexification)**와 **GuSTO(Guaranteed Sequential Trajectory Optimization)**이다. 두 방법은 모두 SCP(sequential convex programming) 계열로, nonconvex problem을 reference trajectory 주변에서 반복적으로 convex subproblem으로 근사하여 푼다.

논문은 이론적 설명에 그치지 않고, 3-DoF rocket landing, quadrotor obstacle avoidance, 6-DoF free-flyer motion planning 예제를 통해 방법론의 실제 적용 흐름을 보여준다. Rocket landing example은 LCvx가 특정 구조에서 global optimality를 회복할 수 있음을 보여주고, quadrotor와 free-flyer example은 SCvx와 GuSTO가 복잡한 obstacle avoidance 및 nonlinear dynamics 문제에서도 실용적으로 작동할 수 있음을 보여준다.

이 논문의 핵심은 “convex optimization으로 모든 nonconvex trajectory problem을 자동으로 해결한다”가 아니다. 더 정확한 해석은, **nonconvex trajectory generation 문제를 convex solver가 잘 다룰 수 있도록 문제 구조를 재구성하는 방법론을 제시한다**는 것이다. 따라서 이 논문은 특정 알고리즘 하나의 성능 논문이라기보다, convex optimization 기반 trajectory generation을 공부하거나 구현하려는 사람에게 문제 설정, 수식 변환, 알고리즘 설계, 실험 해석까지 연결해주는 종합적인 guide에 가깝다.

---

### 1. Introduction

Autonomous system이 실제로 동작하기 위해서는 목표 지점까지의 경로만으로는 충분하지 않다. 드론, 우주선, 자율주행차, free-flyer robot은 모두 자신만의 dynamics와 actuation limits를 가지며, 특정 state나 control constraint를 반드시 만족해야 한다. 따라서 trajectory generation은 “어디로 갈 것인가”를 묻는 문제가 아니라, “어떤 state와 control history를 사용하면 실제 동역학을 만족하면서 목표를 달성할 수 있는가”를 묻는 optimal control problem이다.

논문은 trajectory generation을 다음과 같은 일반적인 optimal control problem으로 출발시킨다.

$$
\begin{aligned}
\min_{u,p,t_f}\quad & J(x,u,p,t_f) \\
\text{s.t.}\quad 
& \dot{x}(t)=f(x,u,p,t), \\
& (x(t),u(t),p,t_f)\in \mathcal{C}(t),\quad \forall t\in[0,t_f], \\
& (x(0),p)\in \mathcal{X}_0, \\
& (x(t_f),p)\in \mathcal{X}_f.
\end{aligned}
$$

이 식에서 $x(t)$는 system state, $u(t)$는 control signal, $p$는 final time scaling 같은 static decision variable, $t_f$는 final time이다. $J$는 mission objective이며, $f$는 dynamics, $\mathcal{C}(t)$는 path constraint, $\mathcal{X}_0$와 $\mathcal{X}_f$는 initial/final condition을 의미한다. 이 식은 논문 전체의 출발점이다. 이후 등장하는 LCvx, SCvx, GuSTO는 모두 이 optimal control problem이 nonconvex일 때 이를 어떻게 convex optimization 기반으로 다룰 것인가에 대한 서로 다른 답이다.

Trajectory generation이 어려운 이유는 이 문제가 원래 continuous-time function space 위의 infinite-dimensional problem이기 때문이다. 실제 계산에서는 temporal discretization을 통해 finite-dimensional optimization problem으로 바꾸지만, 그 결과물도 대부분 nonlinear nonconvex program이 된다. 일반 NLP 방법은 local optimum에 빠질 수 있고, feasible solution을 찾지 못했을 때 실제로 infeasible한 것인지 solver가 실패한 것인지 구분하기 어렵다. 또한 runtime이 안정적으로 제한되지 않아 onboard autonomous system에 그대로 사용하기 어렵다.

이 지점에서 논문은 convex optimization을 도입한다. Convex optimization은 feasible problem에 대해 global optimum을 찾을 수 있고, infeasible problem에 대해 infeasibility certificate를 제공할 수 있으며, polynomial-time complexity를 갖는 solver를 사용할 수 있다. 다만 원래 trajectory generation problem이 자동으로 convex가 되는 것은 아니다. 논문의 핵심 질문은 다음과 같다.

> Nonconvex trajectory generation problem을 어떻게 convex optimizer가 처리할 수 있는 형태로 바꿀 것인가?

이 질문에 대해 논문은 두 가지 큰 방향을 제시한다. 하나는 특정 구조의 nonconvexity를 **lossless하게 convexification**하는 것이고, 다른 하나는 일반 nonconvexity를 **sequential convex approximation**으로 반복 처리하는 것이다. 전자가 LCvx이고, 후자가 SCvx와 GuSTO이다.

---

### 2. Background

#### 2.1. Convex Optimization이 필요한 이유

Convex optimization의 장점은 단순히 “빠르다”에 그치지 않는다. Convex problem에서는 local minimum이 global minimum이며, 많은 problem class에 대해 매우 안정적인 numerical solver가 존재한다. 따라서 safety-critical autonomous system에서 convex optimization은 단순한 수치 기법이 아니라, trajectory generation의 reliability를 확보하기 위한 핵심 computational primitive로 볼 수 있다.

하지만 trajectory generation에서 dynamics는 equality constraint로 들어간다. 일반적인 convex optimization에서 equality constraint는 affine이어야 한다. 따라서 dynamics가 nonlinear이면 그 자체로 convex problem의 구조를 깨뜨린다. 또한 thrust lower bound, obstacle avoidance, pointing constraint처럼 실제 vehicle에서 자연스럽게 등장하는 constraints도 nonconvex인 경우가 많다. 결국 논문이 다루는 핵심은 convex optimization 자체가 아니라, **nonconvex optimal control problem을 convex optimization이 다룰 수 있도록 바꾸는 high-level reformulation strategy**이다.

#### 2.2. LCvx와 SCP의 역할 차이

LCvx와 SCP는 모두 convex optimization을 core solver로 사용하지만, 문제를 다루는 방식은 다르다. LCvx는 특정 nonconvex constraint를 convex relaxation으로 바꾸되, 이 relaxation이 최적해에서 original problem과 동등해지는 것을 보인다. 즉, 조건이 맞으면 original nonconvex problem의 global optimum을 회복할 수 있다.

반면 SCP는 원래 problem을 한 번에 convex problem으로 바꾸지 않는다. 대신 현재 reference trajectory 주변에서 nonlinear dynamics와 nonconvex constraints를 local linearization하고, 그 결과로 얻은 convex subproblem을 반복적으로 푼다. 각 subproblem은 convex solver로 안정적으로 풀 수 있지만, 전체 nonconvex problem에 대해서는 일반적으로 local optimum 또는 stationary point 수준의 보장을 갖는다.

두 방법의 차이는 다음처럼 요약할 수 있다.

| Method | 핵심 아이디어 | 장점 | 주의점 |
|---|---|---|---|
| LCvx | 특정 nonconvexity를 slack variable과 lifting으로 convex relaxation | 조건이 맞으면 original problem의 global optimum 회복 | 적용 가능한 problem structure가 제한적 |
| SCvx | hard trust region과 virtual control을 사용하는 SCP | infeasible initial guess에서도 subproblem feasibility 확보 | virtual control이 최종적으로 0이어야 원래 문제의 feasible solution |
| GuSTO | soft penalty 기반 SCP | continuous-time optimality 분석과 연결 | penalty violation이 남으면 원래 문제에 대해 infeasible할 수 있음 |

이 비교에서 중요한 점은 LCvx와 SCP가 경쟁 관계가 아니라는 것이다. 실제 문제에서는 input lower bound나 pointing constraint처럼 LCvx로 처리 가능한 부분을 먼저 convexify하고, 남은 obstacle avoidance나 nonlinear dynamics를 SCvx/GuSTO로 처리할 수 있다. 논문은 이를 **embedded LCvx** 관점으로 설명한다.

---

### 3. Method

#### 3.1. Problem Formulation

논문에서 trajectory generation problem은 objective, dynamics, path constraint, boundary condition으로 구성된다. 이 문제를 SCP 관점에서 다시 쓰면 convex constraints와 nonconvex constraints를 명시적으로 분리할 수 있다.

$$
\begin{aligned}
\min_{u,p}\quad & J(x,u,p)\\
\text{s.t.}\quad
& \dot{x}(t)=f(t,x(t),u(t),p),\\
& (x(t),p)\in\mathcal{X}(t),\\
& (u(t),p)\in\mathcal{U}(t),\\
& s(t,x(t),u(t),p)\le 0,\\
& g_{\mathrm{ic}}(x(0),p)=0,\\
& g_{\mathrm{tc}}(x(1),p)=0.
\end{aligned}
$$

여기서 $\mathcal{X}(t)$와 $\mathcal{U}(t)$는 convex path constraint set이고, $s(\cdot)$는 nonconvex path constraint를 나타낸다. 이 분리는 SCP의 핵심이다. 이미 convex인 constraints는 그대로 유지하고, nonlinear dynamics와 nonconvex constraints만 reference trajectory 주변에서 선형화한다.

Cost function은 Bolza form으로 표현된다.

$$
J(x,u,p)=\phi(x(1),p)+\int_0^1 \Gamma(x(t),u(t),p)\,dt.
$$

이 식에서 $\phi$는 terminal cost이고, $\Gamma$는 running cost이다. 시간 구간이 $[0,1]$로 normalize되어 있는 이유는 final time이나 time scaling을 parameter $p$ 안에 포함시킬 수 있기 때문이다. 따라서 free-final-time problem도 이 formulation 안에 넣을 수 있다.

#### 3.2. Lossless Convexification (LCvx)

LCvx는 input lower bound처럼 특정 구조의 nonconvex control constraint를 convex relaxation으로 바꾸는 방법이다. 대표적인 nonconvex constraint는 다음과 같다.

$$
\rho_{\min}\le g_1(u(t)),\qquad g_0(u(t))\le\rho_{\max}.
$$

예를 들어 $g_0(u)=g_1(u)=\|u\|_2$이면 이 constraint는 thrust magnitude가 $\rho_{\min}$보다 작아서는 안 되고 $\rho_{\max}$보다 커서도 안 된다는 뜻이다. 문제는 $\rho_{\min}\le\|u\|_2$가 가운데가 비어 있는 annulus 형태의 feasible set을 만들기 때문에 nonconvex라는 점이다.

LCvx는 slack input $\sigma(t)$를 도입해 이 constraint를 다음과 같이 완화한다.

$$
\rho_{\min}\le\sigma(t),\qquad g_0(u(t))\le\rho_{\max},\qquad g_1(u(t))\le\sigma(t).
$$

이 relaxation은 원래 문제보다 feasible set을 넓힌다. 특히 $\|u(t)\|_2<\rho_{\min}$인 input도 relaxed problem에서는 허용될 수 있다. 따라서 단순히 relaxation을 만들었다는 사실만으로는 original problem의 해를 얻었다고 말할 수 없다. LCvx의 핵심은 특정 조건에서 최적해가 relaxed set 내부가 아니라 original feasible input에 대응되는 boundary 위에 놓인다는 점이다.

이를 input norm constraint의 기하학으로 보면 더 직관적이다. 원래 constraint가

$$
\rho_{\min}\le\|u\|_2\le\rho_{\max}
$$

이면, relaxation은

$$
\rho_{\min}\le\sigma,\qquad \|u\|_2\le\min(\sigma,\rho_{\max})
$$

가 된다. 이때 lifted feasible volume은 $(u,\sigma)$-space에서 convex가 된다. 하지만 original feasible input을 회복하려면 solution이 다음 boundary 위에 있어야 한다.

$$
\partial_{\mathrm{LCvx}}\mathcal{V}
=
\{(u,\sigma): \rho_{\min}\le\sigma,\ \|u\|_2=\min(\sigma,\rho_{\max})\}.
$$

즉, LCvx의 목표는 단순히 convex set을 만드는 것이 아니라, convex relaxation의 optimum이 original nonconvex feasible set으로 projection될 수 있음을 보이는 것이다. 논문에서는 이를 Pontryagin maximum principle 기반 조건으로 보장한다.

#### 3.3. Pointing Constraint와 Embedded LCvx

LCvx는 input lower bound뿐 아니라 pointing 또는 tilt constraint에도 사용된다. 예를 들어 input direction이 nominal direction $\hat{n}_u$에서 $\theta_{\max}$ 이상 벗어나지 않아야 한다면 다음 constraint가 등장한다.

$$
\hat{n}_u^\top u(t)\ge \|u(t)\|_2\cos\theta_{\max}.
$$

이 식은 input magnitude와 direction이 결합되어 있어 경우에 따라 nonconvex가 된다. LCvx는 $\|u(t)\|_2$ 대신 slack variable $\sigma(t)$를 사용하여 다음 halfspace 형태로 완화한다.

$$
\hat{n}_u^\top u(t)\ge \sigma(t)\cos\theta_{\max}.
$$

이때도 $g(u)\le\sigma$와 함께 사용되어야 original pointing constraint와 연결된다. 이 구조는 rocket landing에서 thrust pointing constraint를 처리할 때, quadrotor example에서 acceleration direction을 attitude proxy로 사용할 때 중요하게 쓰인다.

실제 trajectory generation problem은 LCvx template에 완전히 들어맞지 않는 경우가 많다. 예를 들어 obstacle avoidance나 nonlinear dynamics가 남아 있으면 LCvx만으로 전체 문제를 convex problem으로 만들 수 없다. 이때 LCvx는 전체 solver가 아니라 **embedded convexification step**으로 사용된다. 즉, LCvx로 처리 가능한 input nonconvexity만 먼저 줄이고, 나머지 nonconvexity는 SCvx 또는 GuSTO가 반복적으로 처리한다.

#### 3.4. Sequential Convex Programming (SCP)

LCvx가 적용되지 않는 일반 nonconvex trajectory generation problem은 SCP로 다룬다. SCP는 현재 reference trajectory $(\bar{x}(t),\bar{u}(t),\bar{p})$ 주변에서 nonlinear dynamics와 nonconvex constraints를 선형화한다.

Dynamics $f$에 대해 다음 Jacobian을 계산한다.

$$
A(t)=\nabla_x f(t,\bar{x}(t),\bar{u}(t),\bar{p}),
$$

$$
B(t)=\nabla_u f(t,\bar{x}(t),\bar{u}(t),\bar{p}),
$$

$$
F(t)=\nabla_p f(t,\bar{x}(t),\bar{u}(t),\bar{p}).
$$

Affine approximation이 reference trajectory를 정확히 지나가도록 offset $r(t)$도 정의한다.

$$
r(t)=f(t,\bar{x},\bar{u},\bar{p})-A(t)\bar{x}(t)-B(t)\bar{u}(t)-F(t)\bar{p}.
$$

이렇게 하면 nonlinear dynamics는 다음 affine dynamics로 근사된다.

$$
\dot{x}(t)=A(t)x(t)+B(t)u(t)+F(t)p+r(t).
$$

이 식은 SCP의 핵심이다. Convex optimization에서는 equality constraint가 affine이어야 하므로, nonlinear dynamics를 convex subproblem 안에 넣기 위해서는 이러한 선형화가 필요하다. Nonconvex path constraint 역시 같은 방식으로 affine inequality로 근사된다.

하지만 linearization은 reference 근처에서만 정확하다. 따라서 subproblem solution이 reference에서 너무 멀어지면 approximation이 원래 problem을 잘못 대표하게 된다. 이를 막기 위해 trust region을 둔다.

$$
\delta x(t)=x(t)-\bar{x}(t),\qquad
\delta u(t)=u(t)-\bar{u}(t),\qquad
\delta p=p-\bar{p}.
$$

$$
\alpha_x\|\delta x(t)\|_q+\alpha_u\|\delta u(t)\|_q+\alpha_p\|\delta p\|_q\le \eta.
$$

여기서 $\eta$는 trust region radius이다. Trust region은 SCP가 한 iteration에서 너무 멀리 움직이지 않도록 제한하며, artificial unboundedness를 막는 역할도 한다.

#### 3.5. SCvx: Virtual Control과 Hard Trust Region

SCvx는 SCP framework에서 artificial infeasibility를 처리하기 위해 **virtual control**을 사용한다. Linearization을 거치면 원래 problem은 feasible하더라도 subproblem은 infeasible해질 수 있다. SCvx는 이를 막기 위해 dynamics에 synthetic input을 추가한다.

$$
\dot{x}=Ax+Bu+Fp+r+E\nu.
$$

여기서 $\nu$는 실제 actuator가 아니라, convex subproblem이 infeasible해지는 것을 막기 위한 virtual control이다. Path constraint와 boundary condition에도 유사한 virtual buffer가 들어간다.

$$
\nu_s\ge Cx+Du+Gp+r'.
$$

Virtual control은 subproblem feasibility를 보장하는 데 유용하지만, 실제 vehicle에는 존재하지 않는 input이다. 따라서 최종 solution에서 virtual control이 0이어야 original problem의 feasible trajectory로 해석할 수 있다. SCvx는 이를 위해 augmented cost에 virtual control penalty를 넣는다.

SCvx의 trust region update는 convex subproblem이 원래 nonlinear problem을 얼마나 잘 예측했는지에 따라 결정된다. 이를 위해 다음 ratio를 사용한다.

$$
\rho=
\frac{
J_\lambda(\bar{x},\bar{u},\bar{p})
-
J_\lambda(x^\star,u^\star,p^\star)
}{
J_\lambda(\bar{x},\bar{u},\bar{p})
-
L_\lambda(x^\star,u^\star,p^\star,\nu^\star,\nu_s^\star)
}.
$$

분자는 실제 nonlinear augmented cost의 감소량이고, 분모는 linearized convex subproblem이 예측한 감소량이다. 따라서 $\rho$는 actual improvement와 predicted improvement의 비율이다. $\rho$가 1에 가까우면 local model이 원래 problem을 잘 근사한 것이고, 너무 작으면 trust region을 줄여야 한다.

#### 3.6. GuSTO: Soft Penalty 기반 SCP

GuSTO는 SCvx와 같은 SCP 계열이지만, artificial infeasibility를 처리하는 방식이 다르다. SCvx가 virtual control과 hard trust region을 사용한다면, GuSTO는 constraint violation과 trust region violation을 soft penalty로 cost에 넣는다.

대표적인 penalty function은 다음과 같다.

$$
h_\lambda(z)=\lambda([z]^+)^2.
$$

또는 smooth approximation으로 softplus 형태를 사용할 수 있다.

$$
h_\lambda(z)=\lambda\sigma^{-1}\log(1+e^{\sigma z}).
$$

여기서 $[z]^+$는 positive part이며, constraint가 위반된 정도만 penalty로 반영한다. $\lambda$는 penalty weight이다. Constraint를 만족하면 penalty가 거의 없고, 위반하면 cost가 커진다.

GuSTO는 convexification accuracy를 다음 normalized error로 평가한다.

$$
\rho=
\frac{
|J_\lambda(x^\star,u^\star,p^\star)-L_\lambda(x^\star,u^\star,p^\star)|+\Theta^\star
}{
|L_\lambda(x^\star,u^\star,p^\star)|+\int_0^1\|\dot{x}^\star\|_2\,dt
}.
$$

여기서 $\Theta^\star$는 nonlinear dynamics와 linearized dynamics의 차이를 측정하는 term이다. SCvx의 $\rho$가 actual improvement와 predicted improvement의 비율이라면, GuSTO의 $\rho$는 cost와 dynamics linearization error를 직접 측정하는 값이다. 따라서 두 알고리즘에서 $\rho$는 같은 이름을 갖지만 해석 방향이 다르다.

#### 3.7. Temporal Discretization

LCvx와 SCP 모두 continuous-time problem에서 출발하지만, 실제 convex solver는 finite-dimensional problem만 풀 수 있다. 따라서 temporal discretization이 필요하다. 논문은 first-order hold(FOH)를 중요한 예로 설명한다.

시간 노드 $0=t_1<t_2<\cdots<t_N=1$를 잡고, 각 interval에서 control을 선형 보간한다.

$$
u(t)=\frac{t_{k+1}-t}{t_{k+1}-t_k}u_k
+\frac{t-t_k}{t_{k+1}-t_k}u_{k+1}.
$$

이를 linearized dynamics에 넣고 적분하면 다음 discrete-time update가 된다.

$$
x_{k+1}
=
A_kx_k+B_k^-u_k+B_k^+u_{k+1}+F_kp+r_k.
$$

이 식이 SCvx와 GuSTO의 convex subproblem에 들어가는 dynamics equality constraint가 된다. Discretization은 단순 구현 세부사항이 아니다. 어떤 discretization을 쓰느냐에 따라 dynamic feasibility, sparsity, runtime, inter-sample constraint violation이 달라진다.

#### 3.8. Algorithm / Pipeline

논문의 전체 방법론은 다음 흐름으로 정리할 수 있다.

1. 먼저 trajectory generation problem을 optimal control problem으로 정식화한다. 이 단계에서 dynamics, objective, path constraints, boundary conditions를 명확히 정의한다.

2. 다음으로 problem 안에 LCvx로 처리 가능한 nonconvex control constraint가 있는지 확인한다. Input lower bound나 pointing constraint처럼 구조가 맞는 경우 slack input $\sigma$를 도입해 convex relaxation으로 바꾼다.

3. LCvx만으로 전체 problem을 해결할 수 없으면 SCP framework로 들어간다. 현재 reference trajectory를 기준으로 dynamics와 nonconvex constraints를 linearization한다.

4. Linearization이 유효한 범위를 제한하기 위해 trust region을 둔다. SCvx는 hard trust region을 사용하고, GuSTO는 trust region violation을 soft penalty로 처리한다.

5. Artificial infeasibility를 처리한다. SCvx는 virtual control을 추가하고, GuSTO는 constraint violation을 penalty cost에 포함시킨다.

6. Continuous-time subproblem을 discretization하여 finite-dimensional convex problem으로 바꾼다.

7. Convex solver로 subproblem을 푼다. 각 subproblem은 convex이므로 solver는 해당 subproblem의 global optimum을 계산할 수 있다.

8. 새 solution이 원래 nonlinear problem을 얼마나 잘 개선했는지 평가한다. SCvx는 actual/predicted improvement ratio를 보고, GuSTO는 cost/dynamics linearization error를 본다.

9. Trust region과 penalty parameter를 update하고, stopping criterion을 만족할 때까지 반복한다.

이 pipeline의 최종 output은 dynamically feasible trajectory $x^\star(t)$, control $u^\star(t)$, 그리고 필요한 경우 time parameter $p^\star$ 또는 final time $t_f^\star$이다. 단, SCvx/GuSTO에서 최종 trajectory가 original problem에 대해 feasible하려면 virtual control 또는 penalty violation이 사라졌는지 확인해야 한다.

---

### 4. Experiments

#### 4.1. Experimental Setup

이 논문의 experiments는 일반적인 ML benchmark와 다르다. 별도의 dataset, training, test split은 존재하지 않는다. 논문은 세 가지 numerical application example을 통해 LCvx, SCvx, GuSTO가 어떤 문제 구조에서 작동하는지 보여준다.

| 항목 | 내용 |
|---|---|
| Dataset | 별도 dataset 없음. Numerical trajectory generation examples 사용 |
| Methods | LCvx, SCvx, GuSTO |
| Examples | 3-DoF rocket landing, quadrotor obstacle avoidance, 6-DoF free-flyer motion planning |
| Baseline | 논문에서 명확한 SOTA baseline 비교는 제시되지 않음. Toy example에서는 maximum principle solution과 비교 |
| Metrics | dynamic feasibility, constraint satisfaction, LCvx tightness, convergence behavior, runtime, subproblem size, inter-sample clipping |

따라서 이 실험은 “어떤 방법이 benchmark에서 가장 높은 score를 얻는가”가 아니라, **각 convex optimization 기반 trajectory generation method가 자신의 목적에 맞게 작동하는가**를 보여주는 demonstration 성격이 강하다.

#### 4.2. LCvx: 3-DoF Fuel-Optimal Rocket Landing

Rocket landing example은 LCvx의 가장 대표적인 적용 사례이다. Powered descent guidance 문제에서는 rocket thrust magnitude에 lower/upper bound가 있고, thrust direction에 pointing constraint가 있으며, glideslope와 velocity constraint도 고려된다. 이 문제는 원래 nonconvex이지만, 논문은 variable substitution과 LCvx relaxation을 통해 convex optimization problem으로 변환한다.

Rocket landing에서 중요한 변수 변환은 다음과 같다.

$$
\xi=\frac{\sigma}{m},\qquad u=\frac{T_c}{m},\qquad z=\ln(m).
$$

여기서 $T_c$는 thrust vector, $m$은 mass, $\sigma$는 slack input이다. 이 변환은 variable-mass dynamics에서 $T_c/m$ 형태로 나타나는 nonlinear term을 다루기 위해 필요하다. 변환 후에는 thrust bound와 mass depletion을 convex optimization이 다룰 수 있는 형태로 정리할 수 있다.

실험적으로 가장 중요한 관찰은 thrust magnitude와 slack variable이 tight하게 일치한다는 점이다. Relaxed problem에서는 $\|T_c(t)\|_2<\rho_{\min}$도 가능하지만, 최적해에서는 그런 input을 사용하지 않는다. 즉, LCvx relaxation이 original nonconvex thrust constraint를 회복한다. 논문은 이 trajectory가 단순 feasible solution이나 local optimum이 아니라, 해당 rocket landing problem에 대한 globally optimal solution이라고 설명한다.

#### 4.3. SCP: Quadrotor Obstacle Avoidance

Quadrotor example은 LCvx만으로는 해결하기 어려운 obstacle avoidance 문제를 SCP로 처리하는 예제이다. Quadrotor는 small and agile vehicle이라는 가정 아래 point-mass double-integrator model로 근사된다.

$$
\ddot{r}(t)=a(t)-g\hat{n}.
$$

여기서 $r(t)$는 position, $a(t)$는 commanded acceleration, $g\hat{n}$은 gravity term이다. Acceleration magnitude lower bound와 tilt constraint는 embedded LCvx로 처리하고, obstacle avoidance constraint는 SCvx 또는 GuSTO를 통해 반복적으로 linearization한다.

Obstacle은 ellipsoidal keep-out zone으로 모델링된다.

$$
\|H_j(r(t)-c_j)\|_2\ge 1,\qquad j=1,\ldots,n_{\mathrm{obs}}.
$$

이 constraint는 obstacle 바깥에 있어야 한다는 뜻이지만, feasible set이 nonconvex이므로 SCP의 linearization 대상이 된다.

실험 결과에서 SCvx와 GuSTO는 모두 infeasible initial guess에서 출발해 obstacle-free trajectory로 수렴한다. 초기 guess는 dynamics와 obstacle constraints에 대해 infeasible하지만, SCP loop를 통해 점차 feasible하고 smooth한 trajectory로 morphing된다. 이 결과는 SCP가 coarse initial guess를 practical trajectory로 바꾸는 데 효과적임을 보여준다.

#### 4.4. SCP: 6-DoF Free-flyer

Free-flyer example은 논문에서 가장 복잡한 application이다. 이 문제는 ISS-like microgravity environment에서 free-flying robot이 장애물을 피하며 이동하는 trajectory를 생성하는 것이다. State는 position, velocity, quaternion attitude, angular velocity를 포함한다.

$$
x=(r_I,v_I,q_{B\leftarrow I},\omega_B)\in\mathbb{R}^{13}.
$$

Control은 inertial thrust $T_I$와 body torque $M_B$로 구성된다.

$$
u=(T_I,M_B)\in\mathbb{R}^6.
$$

Dynamics는 Newton-Euler equation으로 주어진다. 이 예제는 단순 point-mass가 아니라 6-DoF rigid-body motion을 포함하므로, SCvx와 GuSTO가 더 복잡한 nonlinear dynamics 문제에서도 작동하는지를 보여준다.

Flight space는 여러 rectangular room의 union으로 표현되며, 이를 signed distance function(SDF)으로 모델링한다. Exact SDF는 max operator를 포함하므로 non-smooth/nonconvex 구조를 갖는다. 논문은 이를 softmax approximation으로 다룬다.

$$
L_\sigma(v)=\sigma^{-1}\log\sum_{i=1}^{n}e^{\sigma v_i}.
$$

이 softmax approximation은 room SDF들의 max를 smooth하게 근사한다. 다만 이 과정에서 vanishing gradient 문제가 생길 수 있으며, 논문은 slack SDF를 최대화하는 작은 terminal cost를 추가해 이를 완화한다. 이 부분은 SCP formulation에서 gradient behavior가 얼마나 중요한지 보여주는 좋은 예시이다.

Free-flyer 결과에서 SCvx와 GuSTO는 유사한 trajectory를 생성한다. Thrust, torque, attitude, angular rate histories는 constraints 안에서 움직이며, continuous-time rollout이 discrete solution을 잘 통과함으로써 dynamic feasibility를 확인한다. 다만 SDF constraint는 discrete node에서는 만족되지만, node 사이에서 minor inter-sample clipping이 발생한다. 이는 discretization 기반 trajectory optimization의 중요한 한계이다.

#### 4.5. Result Interpretation

이 논문의 experiments는 세 가지 메시지를 전달한다. 첫째, LCvx는 구조가 맞는 rocket landing problem에서 relaxation을 풀어도 original nonconvex input constraint를 회복할 수 있다. 둘째, SCvx와 GuSTO는 obstacle avoidance와 nonlinear dynamics가 포함된 일반 trajectory generation problem에서 infeasible initial guess를 feasible/local trajectory로 변형할 수 있다. 셋째, discretization은 단순한 numerical detail이 아니라 constraint satisfaction과 dynamic feasibility에 직접 영향을 준다.

논문에서 명확한 SOTA baseline comparison이나 formal ablation study는 제시되지 않는다. 따라서 이 실험을 “기존 모든 방법보다 우수함”의 증거로 해석하면 안 된다. 더 적절한 해석은, 이 실험들이 LCvx, SCvx, GuSTO의 적용 방식과 작동 원리를 보여주는 tutorial-style validation이라는 것이다.

---

### 5. Conclusion

이 논문은 convex optimization 기반 trajectory generation을 이해하기 위한 큰 지도를 제공한다. 가장 중요한 출발점은 trajectory generation을 단순 path planning이 아니라, dynamics와 constraints를 만족하는 optimal control problem으로 본다는 점이다. 이 관점에서 nonconvexity는 피할 수 없는 핵심 난점이며, 논문은 이를 LCvx와 SCP라는 두 축으로 정리한다.

LCvx는 특정 구조의 nonconvex control constraint를 slack input과 lifting을 통해 convex relaxation으로 바꾼다. 이때 중요한 것은 relaxation이 단순히 쉬운 근사가 아니라, 조건이 맞으면 original problem의 global optimum을 회복한다는 점이다. 따라서 LCvx는 적용 범위는 좁지만, 성공할 때 매우 강력한 방법이다.

SCvx와 GuSTO는 더 일반적인 nonconvex trajectory generation problem을 다룬다. 두 방법 모두 reference trajectory 주변에서 nonconvex terms를 linearization하고, convex subproblem을 반복적으로 푼다. SCvx는 virtual control과 hard trust region을 사용하고, GuSTO는 soft penalty 기반 formulation을 사용한다. 이 차이는 두 알고리즘의 convergence analysis와 subproblem structure에 반영된다.

이 논문이 보여준 강점은 이론, 수식, 구현, 예제가 균형 있게 연결된다는 점이다. LCvx의 slack variable, SCP의 linearization, trust region, artificial infeasibility 처리, discretization, scaling까지 실제 trajectory optimization을 구현할 때 필요한 요소들을 폭넓게 다룬다. 또한 rocket landing, quadrotor, free-flyer 예제를 통해 각 방법이 어떤 문제 구조에서 사용되는지 보여준다.

다만 이 논문은 tutorial paper이므로, 일반적인 benchmark paper처럼 다양한 baseline에 대한 정량 비교를 제공하지 않는다. SCvx와 GuSTO는 일반적으로 global optimum을 보장하지 않으며, virtual control 또는 penalty violation이 남으면 original problem에 대한 feasible solution으로 볼 수 없다. 또한 temporal discretization 때문에 node 사이 constraint violation이 발생할 수 있다. 실제 적용에서는 node 수, trust region parameter, penalty weight, scaling, constraint margin을 세심하게 조정해야 한다.

결국 이 논문의 핵심 take-home message는 다음과 같다. **Convex optimization은 trajectory generation을 자동으로 해결해주는 마법이 아니라, nonconvex optimal control problem을 잘 구조화했을 때 강력한 solver infrastructure를 제공하는 기반이다.** LCvx는 lossless하게 제거할 수 있는 nonconvexity를 처리하고, SCvx와 GuSTO는 더 일반적인 nonconvexity를 sequential convex approximation으로 다룬다. 이 두 관점을 함께 이해하는 것이 이 논문의 가장 중요한 학습 목표이다.

---

Review by 변정우, Aerospace Engineering Undergraduate Researcher

[Update - Time Log]

YYYY.MM.DD: [Draft]
YYYY.MM.DD: [ver_1]
YYYY.MM.DD: [ver_2]
