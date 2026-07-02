## Differentiable Convex Optimization Layers: CVXPY 기반 Convex Optimization Problem을 Differentiable Layer로 만들기

### 0. 논문 정보 (Reference)

* **Title:** Differentiable Convex Optimization Layers
* **Authors:** Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, J. Zico Kolter
* **Venue / Year:** NeurIPS 2019
* **Keywords:** Differentiable Optimization, Convex Optimization, Disciplined Convex Programming, Disciplined Parametrized Programming, ASA Form, Cone Program, CVXPY, PyTorch, TensorFlow, Implicit Differentiation
* **Code / Project:** CVXPY, cvxpylayers

---

### Abstract

이 논문은 **convex optimization problem을 neural network 또는 differentiable program 안의 layer로 사용할 수 있게 만드는 방법**을 다룬다. 기존에도 optimization problem의 solution을 layer output처럼 사용하고, 그 solution을 통해 gradient를 backpropagation하는 연구는 존재했다. 하지만 실제 사용에는 큰 장벽이 있었다. 사용자가 자신의 convex optimization problem을 직접 QP form, cone program form과 같은 rigid canonical form으로 바꾸어야 했기 때문이다. 논문은 이 과정이 번거롭고, 오류가 나기 쉽고, convex analysis에 대한 지식을 요구한다고 지적한다.

이 문제를 해결하기 위해 논문은 **DPP(Disciplined Parametrized Programming)**와 **ASA form(Affine-Solver-Affine form)**을 제안한다. DPP는 기존 **DCP(Disciplined Convex Programming)**의 subset으로, parameter가 포함된 convex optimization problem이 differentiable layer로 변환될 수 있도록 parameter 사용 방식에 제약을 둔다. ASA form은 DPP-compliant problem의 solution map을 canonicalizer, solver, retriever의 합성으로 분해하는 구조다.

핵심 구조는 다음과 같다.

$$
S = R \circ s \circ C
$$

여기서 $C$는 parameter를 cone problem data로 바꾸는 canonicalizer이고, $s$는 cone solver이며, $R$은 cone solver가 반환한 canonicalized solution에서 원래 optimization problem의 solution을 회수하는 retriever다. 이 논문의 핵심은 $C$와 $R$이 affine map이 되도록 문제 class를 제한하고, 어려운 solver 부분만 cone program differentiation으로 처리한다는 점이다.

구현 측면에서 저자들은 DPP와 ASA form reduction을 CVXPY 1.1에 구현하고, PyTorch와 TensorFlow 2.0에서 사용할 수 있는 differentiable convex optimization layer를 제공한다. 실험에서는 data poisoning sensitivity analysis, stochastic control의 convex approximate dynamic programming, canonicalization time 비교, qpth와의 QP runtime 비교를 통해 제안 방법의 활용 가능성과 실행 시간 경쟁력을 보인다.

이 논문의 의의는 새로운 convex solver를 제안하는 데 있지 않다. 핵심은 **CVXPY-style high-level convex optimization model을 직접 conic form으로 손변환하지 않고도 deep learning framework 안에서 end-to-end differentiable layer로 사용할 수 있게 만든 것**이다.

---

### 1. Introduction

Convex optimization은 control, finance, energy management, signal processing, machine learning 등 다양한 분야에서 구조적 문제를 모델링하는 데 널리 사용된다. Deep learning 관점에서 보면, convex optimization problem을 neural network 안의 layer로 사용할 수 있다면 매우 강한 inductive bias를 줄 수 있다. 일반적인 neural network layer는 explicit function을 계산하지만, optimization layer는 주어진 parameter에 대해 optimization problem을 풀고 그 optimal solution을 output으로 반환한다.

문제는 이러한 optimization layer를 실제로 사용하기 어렵다는 점이다. 기존 differentiable optimization layer 연구들은 특정 optimization problem의 solution을 미분하는 방법을 제시했지만, 많은 경우 사용자가 문제를 solver가 요구하는 canonical form으로 직접 작성해야 했다. 예를 들어 QP layer를 사용하려면 사용자가 objective matrix, linear term, equality constraint matrix, inequality constraint matrix 등을 직접 맞추어야 한다. Cone program을 사용하는 경우에도 마찬가지로 문제를 conic form으로 표현해야 한다.

논문은 이러한 방식이 실제 사용자에게 큰 부담이 된다고 본다. 수학적으로는 convex problem을 알고 있더라도, 이를 solver가 요구하는 low-level matrix form으로 바꾸는 과정은 tedious하고 error-prone하다. 특히 CVXPY 같은 DSL을 사용하면 자연스럽게 작성할 수 있는 문제도, differentiable layer로 사용하려면 다시 canonical form을 직접 만들어야 하는 경우가 많았다.

이 논문이 던지는 핵심 질문은 다음과 같다.

> 사용자가 CVXPY처럼 high-level syntax로 작성한 disciplined convex program을, 자동으로 differentiable optimization layer로 만들 수 있는가?

이 질문은 기존 convex optimization DSL의 역할을 differentiable programming으로 확장하는 문제라고 볼 수 있다. CVXPY는 사용자가 수학적 표현에 가까운 방식으로 problem을 작성하면, 내부에서 canonicalization을 수행해 solver에 넘긴다. 이 논문은 그 과정에 differentiation까지 연결한다. 즉, 사용자는 high-level convex problem을 작성하고, 시스템은 이를 solver form으로 낮춘 뒤, forward solve와 backward gradient computation을 자동으로 수행한다.

추론상으로는 이 논문을 **optimization modeling language와 deep learning autograd 사이의 interface를 만드는 논문**으로 볼 수 있다. 새로운 solver 자체보다 중요한 것은, 사용자가 작성한 convex optimization model을 end-to-end learning pipeline 안으로 자연스럽게 가져오는 구조다.

---

### 2. Background

이 논문의 방법론을 이해하려면 세 가지 배경이 필요하다. 첫째, parameterized convex optimization problem을 solution map으로 보는 관점이다. 둘째, disciplined convex programming(DCP)이 high-level convex expression을 어떻게 검증하고 canonicalize하는지에 대한 이해다. 셋째, cone program이 왜 solver target으로 사용되는지에 대한 이해다.

#### 2.1. Parametrized Convex Optimization Problem과 Solution Map

논문은 먼저 parameter가 포함된 convex optimization problem을 고려한다. 일반적인 형태는 다음과 같다.

$$
\begin{array}{ll}
\text{minimize} & f_0(x;\theta) \
\text{subject to} & f_i(x;\theta) \le 0,\quad i=1,\ldots,m_1, \
& g_i(x;\theta)=0,\quad i=1,\ldots,m_2.
\end{array}
$$

여기서 $x \in \mathbb{R}^n$은 optimization variable이고, $\theta \in \mathbb{R}^p$는 problem을 정의하는 parameter다. $f_i$는 convex function이고, $g_i$는 affine function이다. 즉, $\theta$가 문제의 조건을 정하고, solver는 그 조건 아래에서 최적의 $x$를 찾는다.

이 problem을 differentiable layer로 사용하려면, optimization problem을 단순한 solver call이 아니라 하나의 함수로 보아야 한다. 논문은 이를 solution map이라고 부른다.

$$
S:\mathbb{R}^p \to \mathbb{R}^n,\qquad S(\theta)=x^\star
$$

이 식은 parameter $\theta$가 주어지면 optimal solution $x^\star$가 나온다는 뜻이다. Deep learning 관점에서는 이 solution $x^\star$가 다음 layer 또는 downstream loss의 입력이 된다. 따라서 최종 loss가 $L(x^\star)$라면, 학습에는 $\partial L / \partial \theta$가 필요하다. 결국 이 논문이 계산하고자 하는 것은 solution map $S$의 derivative 또는 backpropagation에 필요한 adjoint derivative다.

다만 논문은 solution map이 single-valued인 경우를 중심으로 다룬다. Convex optimization problem이라고 해서 항상 optimal solution이 하나로 정해지는 것은 아니다. 여러 optimal solution이 존재하면 $S(\theta)$를 일반적인 함수처럼 다루기 어려워진다. 따라서 이 논문은 solution이 unique하게 정해지고 derivative가 존재하는 경우를 기본 전제로 삼는다.

#### 2.2. Disciplined Convex Programming

DCP(Disciplined Convex Programming)는 convex optimization problem을 안전하게 구성하기 위한 grammar다. 사용자는 atom이라고 불리는 기본 함수들을 조합해 objective와 constraint를 작성하고, DCP는 각 expression의 curvature와 monotonicity를 이용해 전체 expression이 convex인지 검사한다.

예를 들어 어떤 함수 $h$가 convex이고 특정 argument에 대해 nondecreasing이라면, 그 argument에는 convex expression을 넣어도 convexity가 보존된다. 반대로 nonincreasing인 argument에는 concave expression을 넣어야 convexity가 유지된다. DCP는 이러한 composition rule을 기반으로, 사용자가 작성한 problem이 convex인지 기계적으로 확인한다.

이 논문에서 DCP가 중요한 이유는 CVXPY 같은 DSL이 바로 DCP를 기반으로 작동하기 때문이다. 사용자는 high-level expression으로 problem을 작성하고, DSL은 DCP 규칙을 통해 convexity를 확인한 뒤, 해당 problem을 solver가 처리할 수 있는 canonical form으로 변환한다.

하지만 DCP만으로는 differentiable layer를 만들기에 충분하지 않다. DCP는 expression이 convex인지 확인하는 grammar지만, parameter가 cone problem data에 어떤 방식으로 들어가는지는 별도로 관리하지 않는다. 이 논문은 이 gap을 메우기 위해 DPP를 도입한다.

#### 2.3. Cone Program

DCP 기반 DSL은 high-level convex program을 solver가 풀 수 있는 cone program으로 canonicalize한다. 논문에서 사용하는 cone program의 기본 형태는 다음과 같다.

$$
\begin{array}{ll}
\text{minimize} & c^T x \
\text{subject to} & b-Ax\in K.
\end{array}
$$

여기서 $A$, $b$, $c$는 problem data이고, $K$는 nonempty, closed, convex cone이다. 원래 problem의 objective와 constraint가 복잡한 convex expression을 포함하고 있더라도, graph implementation과 auxiliary variable을 사용하면 cone program 형태로 변환할 수 있다.

이 변환은 원래 problem의 의미를 바꾸는 것이 아니다. 사용자가 작성한 norm, logistic, quadratic expression 등을 solver가 이해할 수 있는 cone constraint와 linear objective로 번역하는 과정이다. 이 논문에서는 바로 이 canonicalization 과정이 differentiable layer의 일부가 된다.

---

### 3. Method

이 논문의 방법론은 DPP와 ASA form이라는 두 축으로 구성된다. DPP는 어떤 parameterized convex program이 안전하게 differentiable layer로 변환될 수 있는지를 결정하는 grammar이고, ASA form은 실제 forward pass와 backward pass를 구성하는 구조다.

#### 3.1. Problem Formulation

논문의 목표는 solution map $S(\theta)=x^\star$를 미분하는 것이다. 하지만 high-level convex optimization problem을 직접 미분하는 것은 어렵다. 사용자가 작성한 problem은 CVXPY expression tree로 되어 있고, solver는 cone program data를 요구한다. 따라서 중간에 canonicalization이 필요하다.

논문은 이 과정을 다음과 같이 분해한다.

$$
\theta
\xrightarrow{C}
(A,b,c)
\xrightarrow{s}
\tilde{x}^\star
\xrightarrow{R}
x^\star
$$

이 흐름에서 $C$는 parameter $\theta$를 cone problem data $(A,b,c)$로 변환하는 canonicalizer다. $s$는 cone solver로, canonicalized problem을 풀어 solution $\tilde{x}^\star$를 반환한다. $R$은 canonicalized solution에서 원래 problem의 solution $x^\star$를 회수하는 retriever다.

이를 하나의 함수 합성으로 쓰면 다음과 같다.

$$
S = R \circ s \circ C
$$

이 구조가 논문에서 말하는 **ASA form(Affine-Solver-Affine form)**이다. 이름에서 알 수 있듯이 핵심은 solver 앞뒤의 map인 $C$와 $R$이 affine이라는 점이다. 가운데 solver $s$는 일반적으로 nonlinear하지만, 앞뒤 map이 affine이면 canonicalization과 retrieval의 derivative는 매우 단순해진다.

Backward pass는 이 합성 구조에 chain rule을 적용한다.

$$
D^T S(\theta)

=============

D^T C(\theta),
D^T s(A,b,c),
D^T R(\tilde{x}^\star)
$$

이 식은 forward pass와 반대 방향으로 gradient가 흐른다는 뜻이다. 먼저 downstream gradient가 retriever $R$을 거꾸로 통과해 canonicalized solution space로 이동한다. 그 다음 solver derivative를 통해 cone data $(A,b,c)$에 대한 gradient로 바뀐다. 마지막으로 canonicalizer $C$의 adjoint derivative를 통해 original parameter $\theta$에 대한 gradient로 변환된다.

#### 3.2. DPP: Disciplined Parametrized Programming

DPP는 DCP의 subset이지만, 목적은 단순히 convexity를 확인하는 데 있지 않다. DPP의 목적은 parameter가 포함된 convex program이 ASA form으로 reducible하도록 만드는 것이다. 즉, DPP는 parameter-to-data map $C$가 affine이 되도록 parameter 사용 방식을 제한한다.

DPP에서 가장 중요한 차이는 parameter를 constant가 아니라 affine expression처럼 취급한다는 점이다. DCP에서는 parameter가 symbolic constant로 분류되지만, DPP에서는 parameter가 affine으로 분류된다. 이 차이 때문에 parameter가 expression 안에서 어떻게 곱해지는지가 중요해진다.

특히 product atom에서 DPP는 다음과 같은 경우를 허용한다.

1. 한쪽 expression이 constant인 경우
2. 한쪽 expression이 parameter-affine이고, 다른 쪽 expression이 parameter-free인 경우

예를 들어 $F$가 parameter이고 $x$가 variable일 때, $Fx$는 허용될 수 있다. $F$는 parameter-affine이고, $x$는 parameter-free이기 때문이다. 반면 $p_1p_2$처럼 두 parameter가 서로 곱해지는 표현은 DPP가 아니다. 이 경우 problem data가 parameter에 대해 affine하게 유지되지 않기 때문이다.

논문에서 제시하는 대표 예시는 다음 problem이다.

$$
\begin{array}{ll}
\text{minimize} & |Fx-g|_2+\lambda|x|_2 \
\text{subject to} & x\ge 0.
\end{array}
$$

여기서 $x$는 variable이고, $F$, $g$, $\lambda$는 parameter다. 이 problem은 DPP-compliant하다. $Fx$는 parameter-affine expression과 parameter-free expression의 곱으로 볼 수 있고, $\lambda|x|_2$ 역시 nonnegative parameter $\lambda$와 parameter-free convex expression의 곱으로 해석할 수 있기 때문이다.

추론상으로는 DPP를 “CVXPY problem을 differentiable layer로 바꾸기 위한 안전 규칙”으로 이해할 수 있다. DCP가 “이 problem이 convex인가?”를 확인한다면, DPP는 “이 problem이 convex이면서 parameter가 solver data에 affine하게 들어가는가?”를 확인한다.

#### 3.3. Canonicalization and Sparse Affine Map

DPP-compliant problem은 cone program으로 canonicalize된다. 위의 예시 problem은 norm term을 second-order cone constraint로 바꾸어 다음과 같은 형태로 표현할 수 있다.

$$
\begin{array}{ll}
\text{minimize} & t_1+\lambda t_2 \
\text{subject to} & (t_1,Fx-g)\in Q_{m+1}, \
& (t_2,x)\in Q_{n+1}, \
& x\in\mathbb{R}^n_+.
\end{array}
$$

여기서 $t_1$, $t_2$는 norm 값을 표현하기 위해 도입된 auxiliary variable이고, $Q_{m+1}$과 $Q_{n+1}$은 second-order cone이다. 즉, 원래 objective 안에 있던 norm expression을 cone membership constraint로 옮긴 것이다.

이 canonicalized problem을 standard cone program form으로 쓰면, parameter $F$, $g$, $\lambda$가 cone problem data $A$, $b$, $c$의 특정 위치에 들어간다. DPP의 역할은 바로 이 dependency가 affine하게 유지되도록 보장하는 것이다.

논문은 Lemma 1을 통해 canonicalizer map $C$가 sparse matrix와 sparse tensor로 표현될 수 있음을 보인다. Offset 1을 포함한 parameter vector를 $\tilde{\theta}$라고 하면, cone program data는 다음과 같이 계산된다.

$$
c = Q\tilde{\theta}
$$

$$
[A\ b] = \sum_{i=1}^{p+1} R[:,:,i]\tilde{\theta}_i
$$

여기서 $Q$는 sparse matrix이고, $R$은 sparse tensor다. 이때 $R$이라는 기호는 ASA form의 retriever $R$과는 다른 의미로 사용된다. 위 식에서의 $R$은 canonicalization tensor를 의미한다.

이 식의 직관은 단순하다. 처음 한 번 problem structure를 분석하면, 이후 parameter 값이 바뀔 때마다 전체 canonicalization을 다시 수행할 필요가 없다. 대신 미리 만들어 둔 sparse map에 parameter 값을 넣어 cone data를 빠르게 계산하면 된다. 논문에서 CVXPY 1.1의 canonicalization 속도가 크게 개선되는 이유가 바로 여기에 있다.

#### 3.4. Cone Solver and Implicit Differentiation

ASA form에서 가장 어려운 부분은 solver $s$의 derivative다. $C$와 $R$은 affine map이므로 derivative 계산이 비교적 단순하지만, solver $s$는 cone program을 실제로 푸는 nonlinear map이다.

논문은 solver iteration을 모두 unroll하지 않는다. 대신 cone program의 solution이 만족하는 optimality condition을 implicit differentiation한다. Appendix B에서는 primal-dual cone program, homogeneous self-dual embedding, residual map을 이용해 이 과정을 설명한다.

Cone program의 primal-dual form은 다음과 같이 쓸 수 있다.

$$
\begin{array}{ll}
(P)\quad \text{minimize} & c^T x \
\text{subject to} & Ax+s=b, \
& s\in K,
\end{array}
\qquad
\begin{array}{ll}
(D)\quad \text{minimize} & b^T y \
\text{subject to} & A^T y+c=0, \
& y\in K^\ast.
\end{array}
$$

여기서 $x$는 primal variable, $s$는 slack variable, $y$는 dual variable이다. $K^\ast$는 dual cone이다. Optimal solution은 primal feasibility, dual feasibility, complementary slackness를 만족해야 한다. 이러한 조건을 이용하면 solver solution map을 implicit하게 미분할 수 있다.

논문은 homogeneous self-dual embedding에서 problem data $(A,b,c)$를 하나의 skew-symmetric matrix $Q$로 묶는다.

$$
Q =
\begin{bmatrix}
0 & A^T & c \
-A & 0 & b \
-c^T & -b^T & 0
\end{bmatrix}
$$

이 embedding을 사용하면 cone program을 푸는 문제를 residual map $N(z,Q)=0$을 만족하는 $z$를 찾는 문제로 볼 수 있다. 따라서 $Q$가 변할 때 solution $z$가 어떻게 변하는지는 implicit function theorem을 통해 계산할 수 있다.

$$
D s(Q)

======

-\left(D_z N(s(Q),Q)\right)^{-1}
D_Q N(s(Q),Q)
$$

이 식은 solver iteration을 직접 미분하는 것이 아니라, solution이 만족하는 residual equation을 미분한다는 점에서 중요하다. 따라서 unrolled optimization과 달리 특정 iteration 수에 대한 derivative가 아니라, solution map 자체의 derivative를 계산하려는 접근이다. 단, 논문은 non-differentiable point에서는 필요한 linear system이 invertible하지 않을 수 있고, 이 경우 least-squares solution을 heuristic하게 사용한다고 설명한다.

#### 3.5. Solution Retrieval

Canonicalization 과정에서는 원래 variable 외에도 slack variable과 auxiliary variable이 추가된다. 따라서 cone solver가 반환하는 solution은 원래 사용자가 원한 $x^\star$만 포함하지 않는다. 이를 다음과 같이 쓸 수 있다.

$$
\tilde{x}^\star=(x^\star,s^\star)
$$

retriever $R$은 canonicalized solution $\tilde{x}^\star$에서 original problem의 solution $x^\star$를 회수한다.

$$
R(\tilde{x}^\star)=x^\star
$$

논문에서 이 map은 slicing, reshaping, constant scaling 정도로 설명되며, linear map이다. 따라서 backward pass에서 $D^T R$ 역시 간단한 linear operation으로 처리된다.

#### 3.6. Algorithm / Pipeline

논문에는 별도의 algorithm box가 제시되어 있지는 않지만, 전체 방법은 다음 pipeline으로 정리할 수 있다.

1. 사용자는 CVXPY-style syntax로 parameterized convex optimization problem을 작성한다. 이 problem은 DPP-compliant해야 한다.
2. `CvxpyLayer`를 생성할 때 problem structure가 분석되고, canonicalizer $C$와 retriever $R$이 추출된다.
3. Forward pass에서는 parameter $\theta$가 들어오면 $C(\theta)$를 통해 cone problem data $(A,b,c)$가 계산된다.
4. Cone solver $s$가 canonicalized cone program을 풀어 $\tilde{x}^\star$를 반환한다.
5. Retriever $R$이 $\tilde{x}^\star$에서 original solution $x^\star$를 꺼내 layer output으로 반환한다.
6. Downstream loss가 계산되면, backward pass에서는 $D^T R$, $D^T s$, $D^T C$ 순서로 gradient가 전달된다.

이 pipeline은 다음 한 줄로 요약할 수 있다.

$$
\theta
\rightarrow
(A,b,c)
\rightarrow
\tilde{x}^\star
\rightarrow
x^\star
\rightarrow
L(x^\star)
$$

그리고 backward pass는 이 흐름을 반대로 따라간다. 결국 이 논문은 CVXPY problem을 “solve만 가능한 object”가 아니라, “forward와 backward를 모두 가진 differentiable layer”로 만든다.

#### 3.7. Implementation

저자들은 DPP grammar와 ASA form reduction을 CVXPY 1.1에 구현했다. 또한 PyTorch와 TensorFlow 2.0에서 사용할 수 있는 differentiable convex optimization layer를 제공한다. 사용자는 CVXPY problem을 정의한 뒤, parameter와 variable을 지정해 `CvxpyLayer`로 감싸면 된다.

논문에서 보여주는 PyTorch 사용 흐름은 다음과 같이 해석할 수 있다. 먼저 CVXPY로 problem을 정의하고, 해당 problem이 DPP를 만족하는지 확인한다. 그 다음 `CvxpyLayer`를 생성하면 내부적으로 canonicalization이 수행되어 $C$와 $R$이 추출된다. 이후 PyTorch tensor parameter를 layer에 넣으면 forward solve가 수행되고, `.backward()`를 호출하면 parameter에 대한 gradient가 계산된다.

이 구현은 논문의 실용성을 결정하는 부분이다. 이론적으로 solution map을 미분할 수 있다는 것과, 실제 PyTorch/TensorFlow에서 layer처럼 사용할 수 있다는 것은 다르다. 이 논문은 후자까지 구현했다는 점에서 software-methodology contribution이 강하다.

---

### 4. Experiments

이 논문의 실험은 일반적인 deep learning benchmark처럼 accuracy를 비교하는 방식이 아니다. 실험의 목적은 크게 두 가지다. 첫째, 제안한 CvxpyLayer가 실제 gradient-based workflow 안에서 사용될 수 있음을 보여주는 것이다. 둘째, DSL 기반의 general layer가 runtime 측면에서도 실용적일 수 있음을 확인하는 것이다.

#### 4.1. Experimental Setup

실험은 data poisoning attack, stochastic control, canonicalization time comparison, qpth runtime comparison으로 구성된다. 각 실험의 역할은 다음과 같다.

| 실험                    | 검증하려는 것                                                   | 핵심 지표                    |
| --------------------- | --------------------------------------------------------- | ------------------------ |
| Data poisoning attack | optimization solution을 통해 training data까지 gradient가 전달되는가 | test loss gradient       |
| Stochastic control    | SOCP로 정의된 policy를 gradient descent로 학습할 수 있는가             | estimated average cost   |
| Canonicalization time | ASA form이 canonicalization overhead를 줄이는가                 | canonicalization time    |
| qpth comparison       | CvxpyLayer가 specialized QP layer와 비교해 실용적인가               | forward/backward runtime |

Data poisoning 예시는 2D training point 30개와 test point 30개를 사용한다. Stochastic control 예시는 $x\in\mathbb{R}^2$, $u\in\mathbb{R}^3$인 numerical setting에서 수행된다. Runtime 비교에서는 dense QP와 sparse QP를 구성해 qpth와 cvxpylayers를 비교한다.

#### 4.2. Application Example: Data Poisoning Attack

첫 번째 응용 예시는 data poisoning attack이다. 논문은 training data $(x_i,y_i)$로 logistic model을 fit하는 convex optimization problem을 고려한다. 이때 adversary가 training point $x_i$에 작은 perturbation을 줄 수 있다면, test loss를 가장 증가시키는 방향은 test loss의 training data에 대한 gradient로부터 얻을 수 있다.

논문은 adversary가 $|\delta_i|_\infty \le 0.01$인 perturbation을 줄 수 있다고 가정하고, 다음과 같은 policy를 제시한다.

$$
x_i := x_i + 0.01,\mathrm{sign}(\nabla_{x_i}L^{test}(\theta^\star))
$$

이 식은 training data point를 test loss가 증가하는 방향으로 이동시키겠다는 의미다. 중요한 점은 $\nabla_{x_i}L^{test}(\theta^\star)$가 단순히 classifier에 대한 gradient가 아니라, logistic regression optimization problem의 solution을 거쳐 training data까지 전달된 gradient라는 점이다.

Figure 1은 이 gradient를 시각화한다. orange와 blue 점은 서로 다른 class의 training data이고, red dashed line은 training data로 학습된 hyperplane이며, blue solid line은 test loss를 최소화하는 hyperplane이다. 각 data point에 붙은 black line은 test loss의 training data에 대한 gradient 방향을 나타낸다.

이 실험은 CvxpyLayer가 sensitivity analysis에 사용될 수 있음을 보여준다. 다만 이 결과는 data poisoning defense나 attack 자체의 state-of-the-art를 주장하는 실험은 아니다. 2D toy setting에 가까우며, 논문의 목적은 optimization layer를 통해 gradient가 원래 parameter까지 전달됨을 보여주는 데 있다.

#### 4.3. Application Example: Convex Approximate Dynamic Programming

두 번째 응용 예시는 stochastic control이다. 논문은 다음과 같은 stochastic control problem을 고려한다.

$$
\begin{array}{ll}
\text{minimize} & \lim_{T\to\infty}
\mathbb{E}\left[
\frac{1}{T}\sum_{t=0}^{T-1}
|x_t|_2^2+|\phi(x_t)|*2^2
\right] \
\text{subject to} & x*{t+1}=Ax_t+B\phi(x_t)+\omega_t.
\end{array}
$$

여기서 $x_t$는 state, $\phi(x_t)$는 policy, $\omega_t$는 disturbance다. Policy를 직접 함수 공간에서 최적화하는 것은 어렵기 때문에, 논문은 approximate dynamic programming 관점에서 policy를 parameterize한다.

논문에서 policy evaluation은 다음 SOCP를 푸는 것으로 정의된다.

$$
\begin{array}{ll}
\text{minimize} & u^T P u + x_t^T Q u + q^T u \
\text{subject to} & |u|_2 \le 1.
\end{array}
$$

여기서 variable은 control $u$이고, $P$, $Q$, $q$, $x_t$가 parameter로 들어간다. 이 SOCP를 differentiable layer로 만들면, policy parameter $P$, $Q$, $q$에 대해 SGD를 수행할 수 있다.

Figure 2는 gradient descent iteration이 진행됨에 따라 estimated average cost가 감소하는 모습을 보여준다. 논문은 이 numerical example에서 average cost가 약 40% 감소했다고 보고한다. 이는 SOCP-defined policy를 differentiable layer로 넣고, 그 solution을 통해 policy parameter를 학습할 수 있음을 보여주는 결과다.

다만 이 실험은 stochastic control에 대한 포괄적인 benchmark는 아니다. 다양한 system, disturbance distribution, random seed에 대한 상세 비교는 논문에서 명확히 제시되지 않는다. 따라서 이 결과는 “모든 stochastic control problem에서 40% 개선된다”가 아니라, “constrained policy defined by SOCP를 differentiable learning loop에 넣을 수 있다”는 proof-of-concept로 해석하는 것이 적절하다.

#### 4.4. Canonicalization Time Comparison

Table 1은 CVXPY 1.0.23과 CVXPY 1.1의 canonicalization time을 비교한다. 비교 대상은 Section 6에서 사용한 logistic regression과 stochastic control problem이다.

| Problem             |   CVXPY 1.0.23 |      CVXPY 1.1 |
| ------------------- | -------------: | -------------: |
| Logistic regression | 18.9 ± 1.75 ms | 1.49 ± 0.02 ms |
| Stochastic control  | 12.5 ± 0.72 ms | 1.39 ± 0.02 ms |

이 결과는 CVXPY 1.1에서 canonicalization time이 크게 줄었음을 보여준다. Logistic regression에서는 약 12.7배, stochastic control에서는 약 9.0배 정도 빠르다. 논문은 이 speed-up이 canonicalization map $C$를 sparse matrix multiplication으로 계산할 수 있기 때문이라고 설명한다.

이 실험은 DPP와 ASA form의 practical value를 직접 보여준다. DPP는 단순한 문법 제약이 아니라, parameter-to-data mapping을 affine하게 만들어 canonicalization을 빠르게 재사용할 수 있도록 한다. 따라서 training loop에서 parameter가 반복적으로 바뀌더라도, 매번 full DSL canonicalization을 수행할 필요가 없다.

다만 이 table은 canonicalization time만 측정한다. 전체 layer runtime에는 solver forward time, backward time, retrieval time이 포함된다. 따라서 이 결과만으로 “전체 layer가 항상 10배 빨라진다”고 해석하면 안 된다.

#### 4.5. Runtime Comparison with qpth

Figure 3은 PyTorch CvxpyLayer와 qpth를 dense QP 및 sparse QP에서 비교한다. QP는 다음 형태다.

$$
\begin{array}{ll}
\text{minimize} & \frac{1}{2}x^TQx+p^Tx \
\text{subject to} & Ax=b, \
& Gx\le h.
\end{array}
$$

Dense QP에서는 $n=128$, equality constraint 수 $m=0$, inequality constraint 수 $p=128$, batch size 128을 사용한다. Sparse QP에서는 $n=1024$, $m=1024$, $p=1024$, batch size 32를 사용하며, $Q$, $A$, $G$는 각각 1% nonzero를 갖는다.

논문은 dense QP에서 cvxpylayers가 qpth와 경쟁 가능한 runtime을 보인다고 보고한다. Sparse QP에서는 cvxpylayers가 qpth보다 약 5배 빠르다. 그 이유는 cvxpylayers가 sparse operation과 LSQR을 사용해 sparsity를 활용하는 반면, qpth는 sparsity를 충분히 exploit하지 못하기 때문이다.

이 결과는 DSL 기반 general layer가 specialized solver보다 항상 느릴 것이라는 우려를 완화한다. 물론 논문이 모든 problem에서 cvxpylayers가 qpth보다 빠르다고 주장하는 것은 아니다. Discussion에서도 specialized solver가 특정 problem class에서는 더 빠를 수 있다고 언급한다. 따라서 Figure 3은 “general layer도 실용적인 runtime을 가질 수 있다”는 근거로 이해하는 것이 정확하다.

#### 4.6. Ablation / Comparison

논문에는 일반적인 의미의 ablation study가 명시적으로 제시되어 있지는 않다. 즉, DPP를 제거하거나, ASA form을 제거하거나, solver differentiation 방식을 바꾸어 성능을 비교하는 실험은 없다.

다만 Table 1은 ablation에 가까운 역할을 한다. CVXPY 1.0.23과 CVXPY 1.1의 canonicalization time을 비교함으로써, DPP/ASA 기반 canonicalization 개선이 실제 runtime에 어떤 영향을 주는지 보여주기 때문이다. 그러나 이를 엄밀한 component-wise ablation이라고 부르기는 어렵다.

---

### 5. Conclusion

이 논문은 convex optimization problem을 deep learning architecture 안에서 differentiable layer로 사용하기 위한 중요한 software-methodology bridge를 제시한다. 핵심은 사용자가 high-level DSL로 작성한 disciplined convex program을 직접 conic form으로 변환하지 않고도, PyTorch/TensorFlow 안에서 forward solve와 backward gradient computation이 가능한 layer로 만들 수 있다는 점이다.

방법론적으로 가장 중요한 개념은 DPP와 ASA form이다. DPP는 DCP의 subset으로, parameter가 cone problem data에 affine하게 들어가도록 제한한다. 이 제약 덕분에 DPP-compliant problem은 $S=R\circ s\circ C$라는 ASA form으로 표현될 수 있다. 여기서 $C$와 $R$은 affine map이고, solver $s$만 implicit differentiation으로 처리된다. 이 분해는 논문 전체의 수식적 구조이자 구현 구조다.

이 논문이 보여준 실험 결과는 크게 두 방향으로 해석할 수 있다. Figure 1과 Figure 2는 CvxpyLayer가 실제 gradient-based workflow에 들어갈 수 있음을 보여준다. Data poisoning 예시에서는 test loss gradient가 training data까지 전달되고, stochastic control 예시에서는 SOCP-defined policy를 학습해 average cost를 낮춘다. Table 1과 Figure 3은 이 방법이 runtime 측면에서도 practical할 수 있음을 보여준다. 특히 sparse canonicalization과 sparse QP에서의 runtime 이점은 ASA form이 단순한 수식 정리가 아니라 실제 구현상 이득으로 이어짐을 보여준다.

다만 이 논문의 claim을 과장해서 해석하면 안 된다. 모든 DCP problem이 DPP-compliant한 것은 아니며, non-DPP expression은 re-expression이 필요할 수 있다. 또한 general conic solver가 항상 specialized solver보다 빠른 것도 아니다. 논문에서도 OSQP 같은 specialized solver를 backend로 추가하는 방향을 future work로 언급한다. Non-differentiable point에서 least-squares heuristic을 사용하는 부분도 실제 학습 안정성 관점에서는 추가 분석이 필요하다.

그럼에도 이 논문의 가장 큰 의의는 명확하다. Convex optimization layer를 쓰기 위해 사용자가 conic form을 직접 구성해야 했던 장벽을 낮추고, CVXPY-style modeling과 deep learning autograd를 하나의 workflow로 연결했다. 따라서 이 논문은 optimization-based modeling을 neural network 안에 넣고 싶은 연구에서 중요한 기반 논문으로 볼 수 있다.

---

Review by 변정우, Aerospace Engineering Undergraduate Researcher

[Update - Time Log]

YYYY.MM.DD: [Draft]

YYYY.MM.DD: [ver_1]

YYYY.MM.DD: [ver_2]
