## 볼록 최적화와 고도각을 이용한 화성 동력하강 장애물 회피 유도

### 0. 논문 정보 (Reference)

* **Title:** Obstacle avoidance guidance for Mars powered descent using convex optimization and elevation angle
* **Authors:** Duozhi Gao, Youmin Gong, Yanning Guo, Yizheng Xiao, Edoardo Fadda, Paolo Brandimarte
* **Journal:** Acta Astronautica, Vol. 248, 2026, pp. 296–313
* **DOI:** 10.1016/j.actaastro.2026.05.059
* **Official Link:** [Elsevier — Obstacle avoidance guidance for Mars powered descent using convex optimization and elevation angle](https://doi.org/10.1016/j.actaastro.2026.05.059)
* **Keywords:** Powered Descent / Obstacle Avoidance / Convex Optimization / Trajectory Optimization

본문은 원논문과 최종 세미나 발표 자료를 바탕으로 정리하였다. Problem 1–7, Algorithm 1–2, 식 번호, Fig. 및 Table 번호는 모두 **원논문 기준**이며, 별도로 표시한 논의 사항은 결과를 해석할 때 구분해야 할 범위를 다룬다.

---

### 1. Introduction

화성 착륙의 powered descent는 엔진 추력으로 비행체를 감속하면서 목표 위치와 속도에 도달하는 과정이다. 논문은 기존의 평탄한 착륙지 중심 임무에서 crater, canyon, volcanic region과 같은 복잡한 지형으로 탐사 대상이 확대되는 상황을 연구 배경으로 제시한다. 이러한 환경에서는 목표 지점에 정확히 도달하는 것뿐 아니라, 제한된 추력과 연료 안에서 장애물을 회피하는 접근 궤적을 함께 설계해야 한다.

저자들은 기존 유도 방법을 analytical guidance, optimization-based guidance, learning-based guidance로 구분한다. Analytical guidance는 계산이 빠르지만 추력 제한과 복잡한 경로 제약을 동시에 반영하기 어렵고, 연료 최적성이나 파라미터 민감도에 한계가 있다. Learning-based guidance는 빠른 온라인 계산이라는 장점이 있지만 실제 적용을 위해 안정성과 신뢰성을 확인해야 한다. Optimization-based guidance는 동역학과 제약 조건을 문제 안에 직접 넣을 수 있지만, non-convex formulation을 그대로 풀면 계산 부담이 커진다.

이 논문은 그중 **3-DoF convex powered descent guidance**를 출발점으로 삼는다. 기존 lossless convexification으로 최소연료 착륙 문제를 효율적으로 다루되, conventional glide-slope보다 유연한 장애물 모델을 추가하고, 연료 최적화 이후의 접근 형상까지 개선하고자 한다. 특히 장애물과 충돌하지 않는 궤적이라도 장애물 경계를 매우 가깝게 따라갈 수 있으며, 이러한 경로가 착륙 직전의 관측 형상까지 유리하게 만든다고 볼 수는 없다.

따라서 논문의 질문은 두 단계로 나뉜다. 먼저 “장애물 제약을 만족하는 최소연료 궤적을 어떻게 계산할 것인가?”를 다루고, 다음으로 “선택한 연료 범위 안에서 더 수직에 가까운 접근 궤적을 어떻게 만들 것인가?”를 다룬다. 이 두 질문에 각각 대응하는 것이 **Algorithm 1과 Algorithm 2**이며, 마지막에는 elevation-angle integral을 이용해 여러 종단시간 후보 중 최종 궤적을 선택한다.

#### A. Contributions

1. **Iterative Taylor update와 homotopy-based obstacle handling**을 결합한다. 추력 경계의 근사점을 이전 해의 log-mass로 갱신하고, relaxed glide-slope 및 stepwise constraints를 점진적으로 적용한다.
2. **Axis-weighted position integral**을 목적함수로 사용한다. 고도 확보와 수평 위치 감소를 통해 elevation angle과 말기 접근 형상을 간접적으로 개선하면서, 내부 문제의 convex 구조를 유지한다.
3. **Elevation-angle integral에 기반한 terminal-time selection**을 구성한다. 선택한 연료량에서 종단시간을 바꾸어 후보 궤적을 생성하고, 고도 변화에 대해 누적한 고도각을 기준으로 최종 후보를 결정한다.

핵심은 연료 최적화와 관측 형상 개선을 하나의 목적함수로 섞는 것이 아니라, **연료 기준선 계산 → 주어진 연료·시간에서 궤적 형상 개선 → 고도각 적분으로 최종 선택**이라는 역할 분담을 만든 데 있다.

---

### 2. Problem Formulation

#### 2.1. Mathematical Modeling and Nomenclature

착륙선을 중력과 추력을 받는 질점으로 모델링한다. 중력 가속도는 일정하다고 가정하고, 대기 항력과 행성 자전은 무시한다. 좌표계의 원점은 목표 착륙지이며, $x$축은 수직 상방, $y$축은 동쪽, $z$축은 북쪽을 향한다. 따라서 본 논문에서 고도는 $r_x$이고, 수평 위치는 $(r_y,r_z)$이다. (원문 §2.1, Fig. 1)

핵심 변수는 다음과 같다.

| 기호 | 의미 |
|---|---|
| $\mathbf r=[r_x,r_y,r_z]^\top$ | 착륙점 기준 위치 벡터 |
| $\mathbf v=[v_x,v_y,v_z]^\top$ | 속도 벡터 |
| $m$, $\mathbf T$ | 착륙선 질량과 추력 벡터 |
| $\mathbf g_m$, $\lambda$ | 화성 중력 가속도와 연료 소모 계수 |
| $m_{\mathrm{wet}}$, $m_{\mathrm{dry}}$ | 초기 총질량과 기본 문제의 건조질량 하한 |
| $\Gamma$, $\sigma$ | 추력 크기를 다루는 보조변수와 그 질량 정규화 변수 |
| $z=\ln m$, $\mathbf u=\mathbf T/m$ | log-mass와 질량 정규화 추력 |
| $\theta$, $\Psi$ | elevation angle과 그 고도 변화 기반 적분 |
| $\delta$, $\zeta$, $W_\zeta$ | homotopy parameter, 장애물 제약 slack, slack penalty weight |
| $\alpha$ | 수직 방향의 위치 가중치 |

여기서 **log-mass $z$와 위치의 북쪽 성분 $r_z$는 서로 다른 변수**다. 또한 $\mathbf u$는 추력 자체가 아니라 추력을 질량으로 나눈 값이므로 가속도 단위를 가진다.

동역학은 다음과 같이 표현한다.

$$
\begin{aligned}
\dot{\mathbf r}(t)&=\mathbf v(t),\\
\dot{\mathbf v}(t)&=\mathbf g_m+\frac{\mathbf T(t)}{m(t)},\\
\dot m(t)&=-\lambda\lVert\mathbf T(t)\rVert_2.
\end{aligned}
$$

첫 번째 식은 위치 변화, 두 번째 식은 중력과 추력에 의한 가속도, 세 번째 식은 연료 사용에 따른 질량 감소를 나타낸다. 즉, 엔진이 큰 추력을 낼수록 연료가 빠르게 소모되고, 질량이 감소하면서 같은 추력이 만드는 가속도도 달라진다.

연료 소모 계수는 원문 식 (2)에 따라 다음과 같이 둔다.

$$
\lambda=\frac{1}{I_{sp}g_e\cos\phi}.
$$

$I_{sp}$는 specific impulse, $g_e$는 지구 중력 가속도이며, $\phi$는 추력기 배치에 따른 보정에 사용되는 각도다. 여기서 $g_e$는 추진계의 연료 소모 계수를 정의하는 데 쓰이고, 운동방정식의 중력은 $\mathbf g_m$이라는 점을 구분해야 한다.

#### 2.2. Elevation Angle

Elevation angle $\theta$는 착륙점을 기준으로 위치 벡터가 수평면과 이루는 각이다. 수평 거리를

$$
\rho(t):=\sqrt{r_y(t)^2+r_z(t)^2}
$$

로 쓰면, 원문 식 (3)은 다음과 같다.

$$
\theta(t)
=
\arcsin\left(
\frac{r_x(t)}{\sqrt{r_x(t)^2+r_y(t)^2+r_z(t)^2}}
\right)
=
\arctan\left(\frac{r_x(t)}{\rho(t)}\right),
\qquad \rho(t)\ne0.
$$

같은 수평 거리에서 고도가 높거나, 같은 고도에서 수평 거리가 작으면 고도각은 커진다. 논문은 이러한 기하 관계를 이용해 수직에 가까운 접근과 terminal camera field of view의 개선을 유도한다.

다만 $\theta$는 **위치로 정의한 관측 기하 지표**이지, 카메라의 광학적 FoV 크기나 착륙선의 자세각 자체는 아니다. 이하에서는 이러한 차이를 반영해 terminal viewing geometry의 개선으로 해석한다. 또한 정확한 착륙점 $\mathbf r=\mathbf0$에서는 위 비율이 정의되지 않으므로, 착륙 직전의 접근 형상과 착륙점에서의 각도 값을 혼동하지 않아야 한다.

#### 2.3. State and Control Constraints

초기 및 종단 조건은 다음과 같다. 이후 시간 원점은 논문과 같이 $t_0=0$으로 둔다.

$$
\begin{aligned}
\mathbf r(t_0)&=\mathbf r_0,&
\mathbf v(t_0)&=\mathbf v_0,&
m(t_0)&=m_{\mathrm{wet}},\\
\mathbf r(t_f)&=\mathbf r_f,&
\mathbf v(t_f)&=\mathbf v_f,&
m(t_f)&\ge m_{\mathrm{dry}}.
\end{aligned}
$$

수치 실험에서는 목표 위치와 속도를 모두 영벡터로 설정한다. 종단 질량 하한은 추진 과정에서 사용 가능한 연료를 모두 넘겨 쓰지 않도록 하는 조건이다.

추력 크기는 다음 범위에 있어야 한다.

$$
0<T_{\min}\le\lVert\mathbf T(t)\rVert_2\le T_{\max}.
$$

$T_{\min}>0$이므로 하강 중 엔진을 끄는 경우는 허용하지 않는다. 특히 $\lVert\mathbf T\rVert_2\ge T_{\min}$은 원점 부근의 작은 추력을 제외하는 조건으로, 원래 문제의 non-convex성을 만드는 핵심 요소다. (원문 식 (4)–(5))

#### 2.4. Obstacle Avoidance Constraints

논문은 장애물을 세 가지 높이 제약으로 표현한다. 각각의 역할은 **기존 원뿔형 접근 영역 유지**, **충분한 고도에서 제약 완화**, **안전 반경 밖에서 안전 고도 요구**로 정리할 수 있다. (원문 §2.3, Fig. 2)

**A. Conventional glide-slope constraint**

$$
\rho(t)\tan\gamma_{gs}\le r_x(t).
$$

착륙점에서 수평으로 멀리 떨어질수록 더 높은 고도를 요구한다. 이는 second-order cone 형태로 다루기 편리하지만, 실제 장애물보다 충분히 높은 곳에서도 수평 거리에 비례하는 고도 제한을 계속 부과한다.

**B. Relaxed glide-slope constraint**

$$
r_x(t)\ge
\begin{cases}
0,&\rho(t)\le l_1,\\
\bigl(\rho(t)-l_1\bigr)\tan\gamma_{new},&l_1<\rho(t)<l_2,\\
h,&l_2\le\rho(t),
\end{cases}
\qquad
\tan\gamma_{new}=\frac{h}{l_2-l_1}.
$$

착륙점 주변 반경 $l_1$ 안에서는 지면 이상의 고도만 요구하고, $l_1$과 $l_2$ 사이에서는 경사면을 따라 요구 고도를 높인다. 이후에는 요구 고도를 $h$로 제한한다. 여기서 relaxed는 안전을 포기한다는 의미가 아니라, **충분히 높은 영역에서 기존 cone이 부과하던 추가 제한을 완화한다**는 의미다.

**C. Stepwise constraint**

원문 식 (8)의 표기는 다음과 같다.

$$
r_x(t)\ge
\begin{cases}
0,&\rho(t)\le l,\\
h,&l\le\rho(t).
\end{cases}
$$

안전 반경 안쪽에서는 착륙할 수 있지만, 바깥쪽에서는 안전 고도 $h$ 이상을 유지하도록 하는 계단형 모델이다. Relaxed glide-slope의 경사 구간 대신 높이가 급격히 변하는 경계를 사용한다.

원문은 $\rho=l$에서 두 분기의 등호를 중복 표기한다. 따라서 이 식은 논문의 표기대로 제시하되, 실제 코드에서 경계점을 어느 분기로 처리하는지는 별도로 확인해야 한다. 이는 경계의 구현 규칙에 관한 문제이며, 반경 안팎에서 요구되는 높이가 다르다는 모델의 취지와는 구분된다.

Relaxed 및 stepwise 모델은 기존 glide-slope보다 지형에 맞는 접근 영역을 제공하지만, 구간에 따라 달라지는 non-convex 구조 때문에 하나의 SOC constraint로 바로 대체할 수 없다. 이 때문에 뒤에서 homotopy와 반복적인 reference update가 필요해진다.

#### 2.5. Problem 1 and the Relationship between Problems 1–7

**Problem 1은 원래의 최소연료 powered descent 문제**다. 목적함수는 종단 질량 최대화이며, 고정된 초기 질량과 연료 소모 계수 아래에서 추력 크기 적분 최소화와 동등하다.

$$
\min_{\mathbf T(\cdot)}-m(t_f)
\quad\Longleftrightarrow\quad
\min_{\mathbf T(\cdot)}
\int_{t_0}^{t_f}\lVert\mathbf T(t)\rVert_2\,dt.
$$

이 동등성은 다음 질량 관계에서 확인할 수 있다.

$$
m_{\mathrm{wet}}-m(t_f)
=
\lambda\int_{t_0}^{t_f}\lVert\mathbf T(t)\rVert_2\,dt.
$$

문제에는 동역학, 초기·종단 조건, 추력 범위, 앞의 세 장애물 제약 중 해당하는 모델이 함께 들어간다. 추력 하한과 질량에 의존하는 동역학 때문에 원래 formulation은 non-convex optimal control problem이다.

Problem 1–7의 관계를 정리하면 다음과 같다.

| Problem | 역할 | 이전 formulation에서 달라지는 부분 |
|---|---|---|
| P1 | Original fuel-optimal problem | 원래 동역학과 추력·장애물 제약을 포함한 OCP |
| P2 | Relaxed fuel-optimal problem | 추력 크기 완화와 log-mass / mass-normalized variables 도입 |
| P3 | Convexified fuel-optimal problem | 지수형 추력 경계를 Taylor 근사로 변환 |
| P4 | Discretized convex fuel-optimal problem | ZOH 기반으로 이산화한 SOCP |
| P5 | Fuel-optimal problem with improved obstacle avoidance | Taylor reference update, homotopy, slack 적용 |
| P6 | Elevation-angle-based problem with basic obstacle avoidance | P4의 목적함수를 axis-weighted position objective로 변경 |
| P7 | Elevation-angle-based problem with improved obstacle avoidance | P6의 목적함수와 P5의 장애물 처리 구조를 결합 |

즉, 일곱 개의 서로 독립적인 알고리즘을 제안한 것이 아니다. **P1–P4는 기본 문제의 변환, P5는 장애물 처리의 확장, P6–P7은 목적함수 변경에 따른 확장**이다.

---

### 3. Improved Lossless Convexification and Obstacle Avoidance

#### 3.1. Problem 2: Thrust Relaxation and Variable Substitution

먼저 추력 크기를 대신 다룰 scalar auxiliary variable $\Gamma(t)$를 도입한다.

$$
\begin{aligned}
\lVert\mathbf T(t)\rVert_2&\le\Gamma(t),\\
T_{\min}&\le\Gamma(t)\le T_{\max},\\
\dot m(t)&=-\lambda\Gamma(t).
\end{aligned}
$$

원래 추력 벡터에 직접 걸려 있던 크기 하한을 scalar variable의 범위로 옮기는 것이다. 논문은 기존 LCvx 연구를 근거로, 연료 최적화 조건에서 이 완화가 lossless하게 작동한다고 설명한다. 여기서의 relaxation과 뒤에서 사용하는 Taylor approximation은 서로 다른 단계다. (원문 §3.1, 식 (10)–(16))

이후 다음 변수 치환을 사용한다.

$$
z(t):=\ln m(t),
\qquad
\mathbf u(t):=\frac{\mathbf T(t)}{m(t)},
\qquad
\sigma(t):=\frac{\Gamma(t)}{m(t)}.
$$

$\Gamma$는 추력 단위의 보조변수이고, $\sigma$는 질량으로 정규화한 scalar thrust variable이다. 따라서 $\sigma$는 속도나 질량이 아니라 가속도 단위를 가지며, 벡터 $\mathbf u$의 크기를 위에서 제한한다.

치환 후 동역학은 다음과 같이 정리된다.

$$
\begin{aligned}
\dot{\mathbf r}(t)&=\mathbf v(t),\\
\dot{\mathbf v}(t)&=\mathbf g_m+\mathbf u(t),\\
\dot z(t)&=-\lambda\sigma(t),\\
\lVert\mathbf u(t)\rVert_2&\le\sigma(t).
\end{aligned}
$$

특히 log-mass의 미분은

$$
\dot z
=\frac{\dot m}{m}
=-\lambda\frac{\Gamma}{m}
=-\lambda\sigma
$$

가 된다. 이 변환 덕분에 원래 가속도 식의 $\mathbf T/m$과 질량 감소식의 비선형 결합을 affine dynamics로 표현할 수 있다.

초기 및 종단 질량 조건은

$$
z(t_0)=\ln m_{\mathrm{wet}},
\qquad
z(t_f)\ge\ln m_{\mathrm{dry}}
$$

로 바뀌고, 추력 경계는 다음과 같이 남는다.

$$
T_{\min}e^{-z(t)}\le\sigma(t)\le T_{\max}e^{-z(t)}.
$$

이에 따라 **Problem 2**의 목적함수는

$$
\min_{\mathbf u(\cdot),\sigma(\cdot)}-z(t_f)
\quad\Longleftrightarrow\quad
\min_{\mathbf u(\cdot),\sigma(\cdot)}
\int_{t_0}^{t_f}\sigma(t)\,dt
$$

가 된다. 하지만 변수 치환만으로 문제가 완전히 convex해진 것은 아니다. 특히 $\sigma\le T_{\max}e^{-z}$라는 지수함수 아래쪽 영역의 제약이 남아 있으므로, 다음 단계에서 추력 경계를 다시 처리한다.

#### 3.2. Problem 3: Taylor-Based Convexification

원문은 추력 하한과 상한을 서로 다른 차수로 근사한다. 아래 표기는 원문 식 (17)–(18)의 정의를 따른다.

$$
\begin{aligned}
z_l(t)&=\ln\bigl(m_{\mathrm{wet}}-\lambda T_{\min}t\bigr),\\
z_u(t)&=\ln\bigl(m_{\mathrm{wet}}-\lambda T_{\max}t\bigr),\\
d_l(t)&=z(t)-z_l(t),\\
d_u(t)&=z(t)-z_u(t).
\end{aligned}
$$

이를 이용하면 추력 경계는 다음과 같이 된다.

$$
\begin{aligned}
T_{\min}e^{-z_l(t)}
\left(1-d_l(t)+\frac{1}{2}d_l(t)^2\right)
&\le\sigma(t),\\
\sigma(t)
&\le T_{\max}e^{-z_u(t)}\left(1-d_u(t)\right).
\end{aligned}
$$

**추력 하한은 convex quadratic, 추력 상한은 affine 형태**로 다룬다는 것이 핵심이다. 하한 쪽의 $T_{\min}e^{-z}\le\sigma$는 이미 convex epigraph이지만, 논문은 이를 SOCP-compatible한 형태로 표현하면서 지수함수의 곡률을 반영하기 위해 2차식을 사용한다. 반면 상한 쪽은 affine tangent를 이용해 non-convex hypograph를 대체한다.

$z_l$과 $z_u$의 첨자는 각 추력 경계에 사용하는 reference를 구분하는 원문 표기다. 위 정의를 곧바로 “$z_l$이 log-mass의 하한이고 $z_u$가 상한”이라는 순서 관계로 읽어서는 안 된다.

**Problem 3**은 변환된 동역학과 경계조건, conventional glide-slope를 유지하면서 추력 경계를 위 근사식으로 바꾼 연속시간 convex formulation이다. 따라서 이 단계의 의미는 완전히 새로운 유도법을 만드는 것이 아니라, 기존 최소연료 문제를 convex solver가 다룰 수 있는 형태로 정리하는 것이다.

#### 3.3. Problem 4: ZOH Discretization

연속시간 문제를 실제 solver에 입력하기 위해 zero-order hold를 적용한다. 논문의 인덱스에서는 제어 입력이 $k=0,\ldots,N$, 상태가 $k=0,\ldots,N+1$에 정의된다.

$$
\Delta t=\frac{t_f-t_0}{N+1},
\qquad
t_k=k\Delta t,
\qquad
t_0=0.
$$

상태와 중력 입력을 다음과 같이 묶는다.

$$
\mathbf X(t)=
\begin{bmatrix}
\mathbf r(t)\\
\mathbf v(t)\\
z(t)
\end{bmatrix},
\qquad
\mathcal G_m=
\begin{bmatrix}
\mathbf g_m\\
0
\end{bmatrix}.
$$

원문 식 (20)–(24)의 상태공간 표현은

$$
\dot{\mathbf X}
=A_c\mathbf X
+B_c
\begin{bmatrix}
\mathbf u\\
\sigma
\end{bmatrix}
+B_c\mathcal G_m
$$

이며, 행렬은 다음과 같다.

$$
A_c=
\begin{bmatrix}
0_{3\times3}&I_3&0_{3\times1}\\
0_{3\times3}&0_{3\times3}&0_{3\times1}\\
0_{1\times3}&0_{1\times3}&0
\end{bmatrix},
\qquad
B_c=
\begin{bmatrix}
0_{3\times3}&0_{3\times1}\\
I_3&0_{3\times1}\\
0_{1\times3}&-\lambda
\end{bmatrix}.
$$

이에 따라 이산 동역학은

$$
\mathbf X_{k+1}
=A_d\mathbf X_k
+B_d
\begin{bmatrix}
\mathbf u_k\\
\sigma_k
\end{bmatrix}
+B_d\mathcal G_m,
\qquad k=0,\ldots,N
$$

로 표현된다.

$$
A_d=e^{A_c\Delta t},
\qquad
B_d=\int_0^{\Delta t}e^{A_c(\Delta t-\tau)}B_c\,d\tau.
$$

초기 상태, 종단 위치·속도·질량, SOC thrust constraint와 Taylor 근사된 추력 경계도 각 노드에 맞게 적용한다. **Problem 4**의 목적함수는

$$
\min -z_{N+1}
\quad\Longleftrightarrow\quad
\min\lambda\Delta t\sum_{k=0}^{N}\sigma_k
$$

가 된다. 이 formulation은 CVX와 같은 모델링 도구를 통해 구성하고, MOSEK과 같은 convex solver로 풀 수 있는 discrete SOCP다.

여기서 $\lambda\Delta t\sum_k\sigma_k$를 연료 소모량의 kg 값과 혼동하지 않아야 한다. 변환된 질량 동역학을 적분하면

$$
\lambda\Delta t\sum_{k=0}^{N}\sigma_k
=\ln\left(\frac{m_{\mathrm{wet}}}{m(t_f)}\right)
$$

이므로, 이는 연료 소모와 단조 관계를 갖는 **log-mass 기반 목적함수**다. 실제 소모 질량은 $m_{\mathrm{wet}}-m(t_f)$로 계산한다.

#### 3.4. Improved Convexification: Updating the Taylor Reference

Problem 4는 미리 정한 reference mass history를 기준으로 지수형 추력 경계를 근사한다. 실제 해의 log-mass가 이 reference에서 멀어지면 근사 오차가 커질 수 있으며, 논문은 특히 긴 종단시간에서 나타나는 추력 profile의 차이를 지적한다.

이를 개선하기 위해 이전 반복의 해 $\widetilde{\mathbf X}_k$에서 log-mass $\widetilde z_k$를 추출하고, 다음 반복의 Taylor expansion point로 사용한다. (원문 §3.3, 식 (29)–(30))

$$
d_k:=z_k-\widetilde z_k.
$$

갱신된 추력 제약은 다음과 같다.

$$
\begin{aligned}
T_{\min}e^{-\widetilde z_k}
\left(1-d_k+\frac{1}{2}d_k^2\right)
&\le\sigma_k,\\
\sigma_k
&\le T_{\max}e^{-\widetilde z_k}(1-d_k).
\end{aligned}
$$

이는 엔진의 물리적 추력 범위를 바꾸는 과정이 아니다. **현재 구한 질량 이력에 맞춰 추력 경계의 근사점을 다시 잡는 과정**이다. 매 반복에서는 reference가 고정되어 있으므로 convex subproblem을 유지하면서, 반복 사이에서 근사 정확도를 개선한다.

#### 3.5. Problem 5: Homotopy and Slack Relaxation

장애물 제약은 다음 형태로 통합한다.

$$
r_{x,k}\ge F(\mathbf X_k,\delta),
\qquad 0\le\delta\le1.
$$

$F$는 앞에서 정의한 relaxed glide-slope 또는 stepwise 요구 높이에 $\delta$를 적용한 함수다. 예를 들어 relaxed glide-slope의 중간 구간은 $\delta(\rho-l_1)\tan\gamma_{new}$, 외부 구간은 $\delta h$가 된다. Stepwise의 외부 구간 역시 $\delta h$로 표현한다. (원문 식 (31)–(33))

반복 $j$에서 homotopy parameter는

$$
\delta_j=\frac{\min(j,H)}{H}
$$

로 설정한다. 처음에는 쉬운 높이 제약에서 출발하고, 반복이 진행될수록 $\delta=1$인 원래 장애물 제약으로 이동한다. 이는 실제 장애물을 낮춘다는 의미가 아니라, **계산 과정에서 적용하는 요구 높이를 점진적으로 강화한다**는 의미다.

여기서 중요한 것은 reference trajectory의 역할이다. 원문 Algorithm 1의 7번째 단계는 $F(\widetilde{\mathbf X}_k,\delta_j)$를 계산하도록 명시한다. 이를 반복 인덱스를 드러내어 쓰면

$$
F_k^{(j)}:=F(\widetilde{\mathbf X}_k,\delta_j)
$$

가 된다. 즉, 내부 subproblem에서는 이전 궤적으로 계산한 요구 높이를 사용하고, 새로운 해를 얻은 뒤 reference를 갱신한다. 원문의 일반 제약식과 알고리즘의 reference 평가 단계를 함께 읽어야 하며, 움직이는 결정변수에 원래 non-convex 분기식을 그대로 넣었는데 저절로 convex해진다는 의미는 아니다.

하지만 갱신된 높이 제약을 즉시 강제하면 추력 및 종단 조건과 충돌하여 중간 문제가 infeasible해질 수 있다. 이를 완화하기 위해 원문 식 (34)는 slack을 다음 부호로 도입한다.

$$
r_{x,k}\ge F_k^{(j)}+\zeta_k.
$$

**이 표기에서 제약을 완화하는 방향은 음의 $\zeta_k$다.** 예를 들어 요구 높이가 500 m인데 현재 고도가 480 m라면,

$$
480\ge500+(-20)
$$

으로 중간 해를 허용할 수 있다. 따라서 이 식에 $\zeta_k\ge0$을 임의로 추가하면 원문의 완화 의도와 달리 더 강한 고도 제약이 된다.

Slack 사용을 억제하기 위해 **Problem 5**는 다음 목적함수를 사용한다.

$$
\min_{\mathbf X,\mathbf u,\sigma,\zeta}
\lambda\Delta t\sum_{k=0}^{N}\sigma_k
+W_\zeta\lVert\boldsymbol\zeta\rVert_2.
$$

제약은 이산 동역학, 초기·종단 조건, $\lVert\mathbf u_k\rVert_2\le\sigma_k$, 갱신된 Taylor 추력 경계, homotopy-based obstacle constraint다. 첫 번째 목적항은 연료 사용을 줄이고, 두 번째 항은 장애물 제약을 완화하는 정도를 줄인다. 원문은 slack vector의 **2-norm 자체**를 사용하며, 제곱한 2-norm과는 구분한다.

최종 해는 penalty가 작다는 이유만으로 수용하지 않는다. 원래 제약의 만족 여부와 slack tolerance를 함께 확인한다. 따라서 homotopy와 slack은 최종 안전 조건을 제거하는 장치가 아니라, 어려운 제약을 만족하는 해에 도달하기 위한 반복 계산 장치다.

#### 3.6. Algorithm 1

Algorithm 1의 흐름은 다음과 같다.

1. $H$, $\varepsilon_\zeta$를 정하고, $\gamma_{gs}=0$인 Problem 4를 풀어 초기 궤적을 생성한다.
2. $\delta_j$를 증가시키고, 이전 궤적에서 장애물 요구 높이와 Taylor expansion point를 갱신한 뒤 Problem 5를 푼다.
3. 원래 제약의 만족 여부, $\lVert\boldsymbol\zeta\rVert_2\le\varepsilon_\zeta$, $j\ge H+1$을 확인한다. 조건이 충족되지 않으면 새 해를 reference로 삼아 반복한다.

논문에서 사용하는 $H=3$이면 homotopy parameter는 $1/3$, $2/3$, $1$로 증가한다. 종료 조건에는 $j\ge4$도 포함되므로, 원래 높이 제약에 도달한 뒤 추가 갱신을 수행한다. 제시된 사례에서는 네 번의 반복으로 회피 궤적을 얻는다.

Algorithm 1의 출력은 주어진 종단시간에서의 **fuel-optimal obstacle-avoidance baseline**이다. 이후 이 결과를 이용해 fuel–time 관계를 만들고, Algorithm 2가 사용할 연료와 시간의 후보를 정한다.

---

### 4. Elevation-Angle-Based Powered Descent Guidance

#### 4.1. Feasible Fuel–Time Area

Algorithm 1을 여러 종단시간에 대해 실행하면, 각 시간에서 필요한 최소 연료를 얻을 수 있다. 각 시간에서의 최소 연료를 $m_{\mathrm{fuel,min}}(t_f)$로 표기하면, Fig. 3–4의 fuel-optimal curve가 된다.

이 곡선과 함께 구분해야 할 경계는 maximum-thrust fuel, minimum-thrust fuel, maximum fuel carried다. 질량 소모식으로부터 일정 추력에 대응하는 두 직선은

$$
\begin{aligned}
m_{\mathrm{fuel,max\ thrust}}(t_f)
&=\lambda T_{\max}(t_f-t_0),\\
m_{\mathrm{fuel,min\ thrust}}(t_f)
&=\lambda T_{\min}(t_f-t_0)
\end{aligned}
$$

로 표현된다. 반면 탑재 연료 한계는

$$
m_{\mathrm{fuel,carried}}
=m_{\mathrm{wet}}-m_{\mathrm{dry}}
$$

로 결정된다. 기본 설정에서는 $1905-1405=500$ kg이다.

**Maximum-thrust fuel과 maximum fuel carried는 다른 경계**다. 전자는 일정 시간 동안 최대추력을 사용할 때의 소모량이고, 후자는 처음부터 비행체가 가지고 있는 연료의 양이다. 실제 착륙 가능성은 이 scalar bound만으로 결정되지 않으며, 동역학과 종단 조건, 장애물 제약을 만족하는 궤적이 존재해야 한다.

논문은 이러한 관계로부터 fuel–time feasible area를 구성하고, 그 안에서 $(m_{\mathrm{fuel}},t_f)$를 선택한다. Algorithm 1이 주로 최소연료 경계를 찾는 역할이라면, Algorithm 2는 선택한 연료 범위 안에서 접근 궤적의 형상을 개선하는 역할을 한다.

#### 4.2. Problem 6: Axis-Weighted Position Objective

고도각을 직접 최대화하면 위치의 비율과 역삼각함수가 목적함수에 들어간다. 논문은 이를 그대로 최적화하는 대신, 고도각이 커지는 기하학적 방향을 반영한 convex surrogate objective를 사용한다. (원문 식 (37)–(39))

$$J_\alpha
=
\int_{t_0}^{t_f}
\left[
-\alpha r_x(t)
+\lvert r_y(t)\rvert
+\lvert r_z(t)\rvert
\right]dt,
\qquad \alpha>0.
$$

최소화 문제이므로 $-\alpha r_x$는 고도 확보를 선호하게 하고, $\lvert r_y\rvert+\lvert r_z\rvert$는 수평 위치가 착륙점에서 멀어지는 것을 억제한다. 수직 항은 affine이고 수평 항은 convex absolute-value function이므로, 목적함수 자체는 convex하다.

여기서 수평 항은 정확히 $\rho=\sqrt{r_y^2+r_z^2}$가 아니라 **축별 절댓값의 합**이다. 따라서 이 목적함수를 고도각의 정확한 식이나 고도각 적분과 동등한 식으로 읽어서는 안 된다. 고도 확보와 수평 위치 축소라는 방향을 간접적으로 유도하는 목적함수다.

고정된 $t_f$에서 원문은 이산 목적함수를 다음과 같이 쓴다.

$$
\min_{\mathbf X,\mathbf u,\sigma}
\sum_{k=1}^{N}
\left(
-\alpha r_{x,k}
+\lvert r_{y,k}\rvert
+\lvert r_{z,k}\rvert
\right).
$$

**Problem 6**은 Problem 4의 동역학과 conventional glide-slope, Taylor 추력 제약을 유지하고, 연료 최소화 목적함수를 이 위치 기반 목적함수로 바꾼 문제다. 고정된 시간의 Problem 6에서는 공통 양의 계수 $\Delta t$를 생략해도 최소점이 달라지지 않는다. 다만 서로 다른 시간을 비교할 때는 단순히 이 목적값만으로 관측 성능을 평가하지 않으며, 뒤에서 별도의 $\Psi$를 사용한다.

연료 제한은 종단 질량 하한으로 반영한다. 원문은 P6·P7에서 다음 표기를 사용한다.

$$
m_{\mathrm{dry}}=m_{\mathrm{wet}}-m_{\mathrm{fuel}},
\qquad
z_{N+1}\ge\ln\left(m_{\mathrm{wet}}-m_{\mathrm{fuel}}\right).
$$

여기서 $m_{\mathrm{dry}}$는 기본 문제의 물리적 건조질량을 새로 바꾼다는 의미가 아니라, **선택한 fuel budget에 대응하는 terminal-mass lower bound**로 재사용된 표기다. 예를 들어 400 kg의 budget을 선택하면 종단 질량은 1505 kg 이상이어야 한다.

또한 제약은 부등식이므로, formulation 자체가 “선택한 연료량을 반드시 정확히 전부 사용한다”는 등식을 강제하는 것은 아니다. 논문의 fixed-fuel 실험 설명과 실제 optimization에 들어간 fuel-budget constraint를 구분해서 읽어야 한다.

#### 4.3. Problem 7 and Algorithm 2

**Problem 7은 P6의 목적함수와 P5의 장애물 처리 방식을 결합한 문제**다.

$$
\min_{\mathbf X,\mathbf u,\sigma,\zeta}
\sum_{k=1}^{N}
\left(
-\alpha r_{x,k}
+\lvert r_{y,k}\rvert
+\lvert r_{z,k}\rvert
\right)
+W_\zeta\lVert\boldsymbol\zeta\rVert_2.
$$

제약에는 이산 동역학과 초기·종단 상태 조건, 선택한 fuel budget, $\lVert\mathbf u_k\rVert_2\le\sigma_k$, 갱신된 Taylor 추력 경계, reference-based homotopy obstacle constraint가 포함된다. (원문 식 (40), Algorithm 2)

Algorithm 2는 먼저 Algorithm 1을 이용해 fuel–time 영역을 계산하고 $(m_{\mathrm{fuel}},t_f)$를 선택한다. 이후 $\gamma_{gs}=0$인 Problem 6으로 초기 궤적을 만들고, Problem 7을 반복해서 푼다. 반복 과정의 Taylor update, homotopy update, slack norm 및 원래 제약 확인은 Algorithm 1과 같은 구조다.

두 알고리즘의 차이는 내부 반복 형식보다는 **무엇을 최소화하는가**에 있다. Algorithm 1은 연료 기준선을 계산하고, Algorithm 2는 선택한 연료·시간을 유지한 상태에서 위치 기반 목적함수를 최소화한다. 따라서 Algorithm 2가 반환하는 궤적을 모든 조건에서의 새로운 최소연료 해라고 부르는 것은 적절하지 않다.

또한 Algorithm 2 한 번이 전체 시간 탐색을 끝내는 것은 아니다. 한 번의 실행은 선택한 fuel–time pair에 대한 수렴 궤적을 반환하고, 여러 시간 후보를 비교하는 작업은 상위 framework에서 수행한다.

#### 4.4. Effect of the Vertical-Bias Coefficient

$\alpha$는 고도 확보를 얼마나 강하게 선호할지 결정한다. 논문은 stepwise constraint와 400 kg 연료 조건에서 $\alpha=0.1,0.6,1.5,2.5,5,10,20$을 비교하고, 궤적과 고도각, 고도각 적분의 변화를 제시한다. 이 비교의 종단시간은 $t_f=80$ s다. (원문 §4.2, Fig. 6–8)

$\alpha$가 매우 작으면 회피 과정의 고도 확보가 제한되어 장애물 경계에 가까운 궤적이 형성된다. 반대로 $\alpha$가 지나치게 크면 전체 궤적을 위쪽으로 이동시키는 데 연료를 사용하면서, terminal phase에서 형상을 조정할 여유가 줄어들 수 있다. 즉, **고도 가중치를 크게 만드는 것이 말기 접근을 항상 더 좋게 만드는 것은 아니다.**

Fig. 7에서는 작은 $\alpha$가 말기에 비교적 큰 고도각을 갖더라도 회피 구간 전체에서는 불리할 수 있고, 큰 $\alpha$는 오히려 말기 고도각이 작아질 수 있음을 보여준다. 이 때문에 한 시점의 각도나 최대 고도만으로 가중치를 선택하지 않고, 고도각 적분을 함께 비교한다.

논문은 실험적으로 $\alpha\in[0.5,2]$를 적절한 범위로 제시하고, 기본값으로 $\alpha=1$을 사용한다. 이는 검토한 시나리오에서의 선택이며, 모든 초기조건에서 성립하는 보편적 최적 구간이라는 의미는 아니다.

#### 4.5. Elevation-Angle Integral

서로 다른 종단시간을 비교할 때 단순한 시간 적분 $\int\theta\,dt$를 사용하면 비행시간의 영향이 함께 들어간다. 논문은 이를 피하기 위해 고도 변화에 대해 elevation angle을 누적한다. 원문 식 (41)의 시간 매개화 표현은 다음과 같다.

$$
\Psi
=
\int_{t_0}^{t_f}
\theta(t)\bigl(-v_x(t)\bigr)\,dt,
\qquad
 d\bigl(-r_x(t)\bigr)=-v_x(t)\,dt.
$$

이산 평가는 원문 식 (42)를 따른다.

$$
\begin{aligned}
\theta_k
&=\arcsin\left(
\frac{r_{x,k}}
{\sqrt{r_{x,k}^2+r_{y,k}^2+r_{z,k}^2}}
\right),\\
\Psi
&\approx\sum_{k=1}^{N}
\theta_k\bigl(-v_{x,k}\bigr)\Delta t.
\end{aligned}
$$

논문은 이산 합 자체를 $\Psi$로 표기한다. 위 근사 기호는 연속 적분과 수치 합을 구분하기 위한 것이며, 합의 범위와 평가식은 원문을 따른다. 특히 $k=1,\ldots,N$이므로 정확한 착륙점인 $N+1$번째 상태에서 고도각을 계산하는 식은 아니다.

하강 구간에서는 $v_x<0$이므로 고도각이 양의 가중치로 누적된다. 반대로 상승 구간이 존재하면 $-v_x<0$이 된다. 따라서 이 식은 **고도 변화의 부호를 반영하는 적분**이며, $\lvert v_x\rvert$로 바꾸어 총 이동 고도에 대해 누적한 지표와는 다르다.

원문의 그림은 $\theta$를 degree로 평가하여 $\Psi$를 degree·m 단위로 표시한다. Radian으로 계산하면 적분의 수치 크기는 달라지지만, 동일한 각도 단위를 일관되게 사용하는 것이 우선이다.

무엇보다 **$J_\alpha$는 궤적 생성용 surrogate이고, $\Psi$는 생성된 후보의 평가 지표**다. Algorithm 2가 $\Psi$를 직접 최대화하는 convex optimization을 푸는 것은 아니다.

#### 4.6. Fixed Terminal Time versus Fixed Fuel

종단시간을 고정하면 연료 사용을 늘려 궤적 형상을 조정할 수 있다. 논문은 $t_f=80$ s에서 연료량을 바꾸어, 연료 최적 궤적보다 위쪽으로 이동하고 수직 접근에 가까워지는 결과를 제시한다. 다만 같은 양의 추가 연료에 대한 개선 폭은 점차 줄어든다. 따라서 이 사례의 의미는 무조건 많은 연료를 사용하는 것이 유리하다는 것이 아니라, 최소연료 기준선에서 조금의 여유를 활용했을 때 얻는 형상 개선을 평가한다는 데 있다. (원문 §4.3, Fig. 9–12)

반대로 연료를 400 kg으로 고정하고 종단시간을 바꾸면, 가능한 시간 범위의 양 끝에서는 장애물 경계에 가까운 궤적이 형성되고, 중간 시간에서는 더 큰 회피 여유를 갖는 궤적이 나타난다. 이때 Fig. 15의 $\Psi$ 최대점과 Fig. 16의 위치 기반 목적함수 최소점은 일치하지 않는다. (원문 §4.4, Fig. 13–16)

이 비교는 **더 짧은 시간, 더 긴 시간, 더 작은 surrogate objective 중 어느 하나만으로 최종 궤적을 선택할 수 없다는 점**을 보여준다. 두 목적의 차이는 오차가 아니라, 생성에 사용한 위치 기반 시간 적분과 평가에 사용한 고도각의 고도 변화 적분이 서로 다른 지표이기 때문에 발생한다.

#### 4.7. Terminal-Time Selection Framework

원문 Fig. 17의 전체 절차는 다음과 같이 정리할 수 있다.

$$
(m_{\mathrm{fuel}},t_f)
\xrightarrow{\text{Algorithm 2}}
\mathbf X^*(\cdot)
\longrightarrow
\theta_k
\longrightarrow
\Psi.
$$

주어진 연료 조건에서 가능한 종단시간 범위를 구하고, 그 범위 안의 시간 후보마다 Algorithm 2를 실행한다. 각 궤적의 $\Psi$를 계산한 뒤 가장 큰 값을 갖는 후보를 선택한다. 이 과정을 후보 집합 $\mathcal T(m_{\mathrm{fuel}})$에 대한 선택식으로 쓰면 다음과 같다.

$$
t_f^*(m_{\mathrm{fuel}})
\in
\underset{t_f\in\mathcal T(m_{\mathrm{fuel}})}{\operatorname{arg\,max}}
\;\Psi\bigl(\mathbf X^*(\cdot;m_{\mathrm{fuel}},t_f)\bigr).
$$

$\mathcal T$는 Fig. 17의 시간 증가 절차를 설명하기 위해 사용한 표기다. 내부의 $\mathbf X^*$는 각 시간에서 위치 기반 목적함수로 생성한 궤적이고, 외부의 선택은 고도각 적분에 따라 수행된다.

따라서 논문에서 말하는 optimal terminal time은 제시된 framework가 탐색한 후보를 기준으로 해석해야 한다. Fig. 15–16의 별표 역시 도시된 후보에서의 최댓값 또는 최솟값에 해당한다. 이것을 원래 non-convex 자유 종단시간 문제 전체에 대한 전역 최적성 증명과 동일하게 볼 수는 없다.

---

### 5. Numerical Experiments

#### 5.1. Simulation Setup

논문은 MATLAB R2023b, CVX, MOSEK을 이용해 제안 방법을 구현했다. 계산 환경은 Intel Core i9-13900HX 2.20 GHz와 32 GB RAM이다. 기본 설정은 다음과 같다. (원문 §5.1, Table 1–2)

| 항목 | 값 |
|---|---|
| 초기 질량 $m_{\mathrm{wet}}$ | 1905 kg |
| 기본 건조질량 $m_{\mathrm{dry}}$ | 1405 kg |
| 기본 탑재 연료 한계 | 500 kg |
| 최대 추력 $T_{\max}$ | 13258.17 N |
| 최소 추력 $T_{\min}$ | 4971.82 N |
| 연료 소모 계수 $\lambda$ | $5.09\times10^{-4}$ s/m |
| 중력 $\mathbf g_m$ | $[-3.7114,0,0]^\top$ m/s$^2$ |
| 초기 위치 $\mathbf r_0$ | $[1500,0,1500]^\top$ m |
| 초기 속도 $\mathbf v_0$ | $[-75,0,70]^\top$ m/s |
| 목표 위치·속도 | $\mathbf r_f=\mathbf0$, $\mathbf v_f=\mathbf0$ |
| 기본 종단시간 | 80 s |
| $\alpha$, $W_\zeta$ | $1$, $10^3$ |
| $l_1$, $l_2$, $l$, $h$ | 100 m, 500 m, 500 m, 500 m |
| Homotopy step 수 $H$ | 3 |
| Slack tolerance $\varepsilon_\zeta$ | $10^{-6}$ |

초기조건과 종단시간은 실험 목적에 따라 변경된다. 따라서 이후 결과를 읽을 때 기본 설정의 stepwise 사례와, 초기조건을 바꾼 relaxed glide-slope 사례를 구분해야 한다.

#### 5.2. Algorithm 1: Thrust Approximation and Computational Performance

먼저 conventional glide-slope angle을 $10^\circ$로 두고, 종단시간 78, 98, 118, 138 s에 대해 기존 convex method와 improved method의 추력 이력을 비교한다. Fig. 18에서 기존 방법은 시간이 길어질수록 최대·최소 추력에 가까운 profile에서 벗어나는 차이가 커진다. 반면 Taylor expansion point를 갱신한 방법은 이러한 차이를 줄인다.

이 결과는 단순히 제어 입력을 더 매끄럽게 만들었다는 뜻이 아니다. **미리 정한 질량 reference와 실제 질량 이력의 차이를 반복적으로 줄여 추력 경계 근사를 개선했다**는 점이 중요하다.

Fig. 18(d)에 대응하는 원문 Table 3의 비교는 다음과 같다.

| 비교 항목 | Algorithm 1 | Algorithm 2 | Conventional convex method | GPOPS |
|---|---:|---:|---:|---:|
| Average CPU time (s) | 0.06 | 0.06 | 0.03 | 1.12 |
| Fuel consumption (kg) | 505.49 | 505.49 | 505.51 | 505.484 |
| Step-shaped obstacle avoidance | Yes | Yes | No | Yes |
| Improve field of view | No | Yes | No | Yes |

표의 앞 두 행은 Fig. 18(d)의 계산 결과이고, 뒤 두 행은 논문이 정리한 방법별 기능 비교다. 제안 방법은 기존 convex method보다 계산시간이 늘지만, 이 사례에서는 GPOPS보다 낮은 CPU time으로 비슷한 연료값을 얻는다. 또한 Algorithm 2는 같은 표에서 관측 형상 개선 기능을 가진 방법으로 구분된다.

다만 **0.06 s는 이 비교 시나리오의 보고값**이다. Fuel–time 영역 생성과 여러 종단시간 후보의 계산을 포함한 전체 framework의 실행시간이 모두 0.06 s라는 의미는 아니다.

또한 Table 3의 연료값은 원문 그대로 옮겼다. 약 505.49 kg은 Table 1에서 계산한 기본 탑재 연료 500 kg보다 크다. 제공된 자료에는 이 비교에서 질량 하한을 별도로 변경했는지 명확한 설명이 없으므로, 이를 기본 500 kg 제한까지 동시에 만족한 결과로 해석해서는 안 된다.

장애물 처리 자체는 Fig. 19–20에서 확인한다. Stepwise와 relaxed glide-slope 사례 모두 네 번의 반복으로 회피 궤적을 얻으며, 논문은 최종 slack norm이 tolerance보다 충분히 작아졌다고 보고한다. 이 부분은 homotopy가 중간 완화 문제에서 원래 장애물 조건을 만족하는 해로 진행한다는 것을 보여준다.

#### 5.3. Algorithm 2: Obstacle Clearance at the Same Fuel Level

Relaxed glide-slope 실험에서는 초기조건을 다음과 같이 변경한다.

$$
\mathbf r_0=[1500,0,500]^\top\ \mathrm{m},
\qquad
\mathbf v_0=[-75,0,-100]^\top\ \mathrm{m/s}.
$$

Algorithm 1의 $t_f=80$ s 사례에서는 365.18 kg을 사용하며, $t_f=100$ s로 바꾸면 409.12 kg을 사용한다. 그러나 Fig. 20과 Fig. 21을 비교하면, 연료를 더 쓴 두 번째 궤적도 장애물 회피 형상이 크게 개선되지는 않는다. 이는 **최소연료 문제를 더 긴 시간에서 다시 푸는 것만으로 관측 형상이 개선되는 것은 아니라는 사례**다.

이후 Algorithm 2를 적용하면 같은 연료 수준에서도 다른 접근 형상을 얻을 수 있다. 365.18 kg 조건의 Fig. 23–26에서는 개선이 나타나지만 그 폭이 비교적 작다. 반면 409.12 kg 조건의 Fig. 27–30에서는 종단시간을 바꾸어 생성한 후보 중 장애물과의 여유를 더 크게 확보하고, 착륙 직전에 수직에 가까운 접근을 보이는 궤적이 나타난다. (원문 §5.2–5.3)

Fig. 21의 기준 궤적과 Fig. 27의 후보들은 409.12 kg이라는 같은 연료 수준을 바탕으로 비교되지만, 종단시간과 목적함수는 달라진다. 따라서 결과를 **같은 연료와 같은 시간의 동일 문제에서 모든 성능이 동시에 향상되었다**고 설명하면 비교 조건을 놓치게 된다.

정확한 해석은 주어진 연료를 더 많이 쓰는 대신, **그 연료를 어느 시간과 어떤 궤적 형상에 배분할지 바꾸어 회피 여유와 말기 관측 형상을 개선했다**는 것이다. 또한 적은 연료 조건에서 개선 폭이 제한적이었다는 결과도 함께 고려해야 한다.

#### 5.4. Fuel–Time Trade-off and Final Trajectory Selection

Fig. 31은 여러 연료 수준에서 종단시간에 따른 $\Psi$를 비교하고, Fig. 32는 같은 후보들의 위치 기반 목적값을 비교한다. 제시된 사례에서 연료가 고정되면 $\Psi$가 가장 커지는 시간 후보가 나타나며, 연료 수준을 바꾸면 그 최대점의 위치도 달라진다.

이 최대점들을 fuel–time plane에 표시한 것이 Fig. 33의 **optimal integral of elevation angle curve**다. 반면 위치 기반 목적함수의 최소점을 연결하면 별도의 **optimal objective value curve**가 만들어진다. 두 곡선은 전체 경향이 비슷하지만 일치하지 않는다.

이 결과는 Algorithm 2의 목적값만으로 상위 시간 선택을 대신해서는 안 된다는 앞선 설명을 수치적으로 보여준다. 같은 연료 조건에서도 시간 선택에 따라 관측 기하가 달라지고, 동일한 $\Psi$ 수준을 만드는 연료·시간 조합이 여러 개 존재할 수 있다. 논문은 이런 경우 더 적은 연료를 쓰는 조합을 선택할 수 있다고 설명한다. (원문 §5.4, Fig. 31–33)

따라서 Fig. 33은 단순한 최적 궤적 그림이라기보다, **fuel budget을 선택했을 때 어떤 terminal-time candidate가 관측 형상에 유리한지 보여주는 설계 관계**다. 다만 제시된 곡선의 단봉 형태는 해당 수치 사례의 관찰이며, 모든 초기조건에서 같은 형태가 보장된다는 일반 정리로 제시된 것은 아니다.

#### 5.5. Different Initial Conditions and 3-DoF Cases

마지막 검증은 특정 평면 초기조건에만 결과가 의존하는지 확인하는 단계다. 논문은 Table 4의 평면 사례와 Table 5의 3-DoF 사례를 사용해 Algorithm 1과 elevation-angle-based optimal guidance를 비교한다.

원문 Table 5의 초기조건은 다음과 같다.

| Case | 초기 위치 $\mathbf r_0$ (m) | 초기 속도 $\mathbf v_0$ (m/s) | 표에 제시된 $t_f$ (s) |
|---|---|---|---:|
| 1 | $[1500,2000,500]^\top$ | $[-75,-20,10]^\top$ | 80 |
| 2 | $[1500,500,2000]^\top$ | $[-75,0,-30]^\top$ | 80 |
| 3 | $[1500,1000,2000]^\top$ | $[-75,20,20]^\top$ | 80 |
| 4 | $[1500,2000,1000]^\top$ | $[-75,-10,10]^\top$ | 80 |
| 5 | $[1500,2000,2000]^\top$ | $[-75,10,-10]^\top$ | 80 |

Fig. 35에서 점선은 fuel-optimal trajectory, 실선은 elevation-angle-based optimal obstacle-avoidance trajectory다. 두 방법 모두 장애물을 회피하지만, 논문은 같은 연료 소모를 유지한 비교에서 고도각 기반 방법이 더 유리한 말기 관측 형상과 회피 성능을 보인다고 보고한다. Fig. 35의 3차원 궤적과 투영도에서도 목표점에 가까워질수록 수평 이동이 줄어드는 접근 형상이 나타난다.

여기서 Table 5의 기준 시간 80 s와 상위 framework가 탐색하는 최종 시간은 구분해서 읽어야 한다. 같은 연료를 사용했다는 설명만으로 두 방법이 모든 사례에서 종단시간까지 같았다고 확대해서 해석할 수는 없다.

또한 원문은 이 절을 robustness 검증이라고 부르지만, 직접 제시된 근거는 **여러 지정 초기조건에서의 수치 궤적 비교**다. 이를 navigation error, 외란, 모델 불확실성이 포함된 확률적 안전성이나 대규모 Monte Carlo 성공률의 검증으로 확장할 수는 없다.

---

### 6. Conclusion

본 논문은 화성 powered descent에서 장애물 제약을 처리하는 convex optimization 구조와, 연료 제약 안에서 접근 형상을 개선하는 위치 기반 목적함수를 결합했다. 전체 내용을 세 단계로 정리하면 다음과 같다.

1. **Algorithm 1:** 추력 경계의 Taylor reference를 반복 갱신하고 homotopy와 slack을 이용해 장애물 제약을 처리하여 fuel-optimal baseline을 계산한다.
2. **Algorithm 2:** 선택한 fuel budget과 terminal time에서 axis-weighted position objective를 최소화하여 vertical-biased trajectory를 생성한다.
3. **Overall framework:** 각 시간 후보에서 얻은 궤적의 elevation-angle integral $\Psi$를 비교하여 최종 terminal time과 trajectory를 선택한다.

수치 결과는 이 역할 분담의 필요성을 보여준다. 추력 근사 갱신은 긴 종단시간에서 기존 근사의 차이를 줄이고, 위치 기반 목적함수는 연료 최소화만으로는 얻기 어려운 접근 형상을 만든다. 또한 동일 연료 수준에서도 시간을 적절히 선택하면 장애물과의 여유 및 착륙 직전의 관측 기하를 개선할 수 있다.

다만 결과를 해석할 때는 다음 범위를 구분해야 한다.

**첫째, FoV 개선은 3-DoF 궤적 기하에 기반한 결과다.** 현재 모델에는 자세, 카메라 광축, 실제 영상의 가시 영역이나 touchdown contact dynamics가 포함되지 않는다. 따라서 near-vertical approach가 관측과 착륙 안전에 유리한 방향이라는 설명과, 실제 카메라 성능 또는 전도 위험이 검증되었다는 주장은 다르다. 원논문이 제시한 향후 연구도 6-DoF powered descent로의 확장이다.

**둘째, convex subproblem의 해와 원래 non-convex 문제 전체의 보장은 구분해야 한다.** 논문은 LCvx, Taylor approximation, reference-based obstacle handling을 결합한다. 특히 연료 최적화에서 설명한 lossless relaxation의 성질을 목적함수가 변경된 P6·P7에 아무 조건 없이 확대해서 읽어서는 안 된다. 원문 Algorithm 1·2에 포함된 원래 제약 확인은 이러한 구분에서 중요한 절차다.

**셋째, 관측 형상과 연료·시간 사이에는 trade-off가 남는다.** $\alpha$를 크게 하거나 연료를 더 쓰거나 시간을 늘리는 것만으로 결과가 항상 좋아지지는 않는다. 또한 내부 surrogate objective의 최소점과 외부 $\Psi$의 최대점도 다를 수 있으므로, 생성 기준과 평가 기준을 분리해서 읽어야 한다.

결국 이 논문의 공학적 의의는 **최소연료 궤적을 얻는 문제에서 한 단계 더 나아가, 주어진 연료 안에서 착륙 접근 형상을 어떻게 개선하고 선택할지 구조화했다는 점**에 있다. Convex optimization의 계산상 장점을 활용하면서, 장애물 회피와 말기 수직 접근을 연결하는 framework를 제시한 연구로 정리할 수 있다.

---

### References

* Gao, D., Gong, Y., Guo, Y., Xiao, Y., Fadda, E., and Brandimarte, P., “Obstacle avoidance guidance for Mars powered descent using convex optimization and elevation angle,” *Acta Astronautica*, Vol. 248, 2026, pp. 296–313. [DOI: 10.1016/j.actaastro.2026.05.059](https://doi.org/10.1016/j.actaastro.2026.05.059)
* 변정우, *Obstacle Avoidance Guidance for Mars Powered Descent Using Convex Optimization and Elevation Angle*, 정기 논문 세미나 발표 자료, 2026.10.02, 최종 PPT.

---

**Review by 변정우, Aerospace Engineering Undergraduate Researcher**  
**[Update - Time Log]**
* 2026.10.03: [ver_1] 원논문과 최종 세미나 PPT를 바탕으로 Problem 1–7, Algorithm 1–2, 수치 결과 및 해석 범위를 정리
