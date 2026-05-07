## SAFE: Sparse 및 Flat Minima 탐색을 통한 Pruning 성능 향상  
  
### 0. 논문 정보 (Reference)  
* **Title:** SAFE: Finding Sparse and Flat Minima to Improve Pruning  
* **Authors:** Dongyeop Lee, Kwanhee Lee, Jinseok Chung, Namhoon Lee  
* **Conference:** ICML 2025 (Spotlight Poster)  
* **Proceedings:** Proceedings of the 42nd International Conference on Machine Learning, PMLR Vol. 267, 2025  
* **arXiv:** 2506.06866  
* **OpenReview:** https://openreview.net/forum?id=10l1pGeOcK  
* **Open Source:** https://github.com/LOG-postech/safe-jax / https://github.com/LOG-postech/safe-torch

---

### Abstract

이 논문은 pruning을 다른 시각으로 접근하여 해석하였다. 이를 단순히 파라미터 수를 줄이는 문제로 다루지 않고, sparsity 과 flatness를 동시에 만족하는 해를 찾는 constraint optimization problem으로 다시 정의한다. 기존 pruning은 높은 sparsity에서 성능 저하가 발생하였는데, 그 원인을 단순한 용량 감소뿐 아니라 sharp한 손실 지형 위의 sparse solution에서 찾고자 한다. 따라서 목표는 단순히 많은 가중치를 0으로 만드는 것이 아니라, 작은 교란에도 성능이 급격히 나빠지지 않는 sparse subnetwork 를 찾는 데에 있다.

이를 위해 논문은 다음과 같은 sharpness-aware sparsity-constrained optimization 문제를 제안한다.

$$
\min_{\|x\|_0 \le d} \max_{\|\epsilon\|_2 \le \rho} f(x+\epsilon)
$$

여기서 바깥 minimization은 희소한 해를 찾는 과정이고, 안쪽 maximization은 현재 해 주변의 작은 perturbation 중 가장 손실을 크게 만드는 방향을 고려함으로써 flat minima를 유도한다. 이후 이 문제를 augmented Lagrangian과 ADMM 관점으로 풀어내며, 연속적으로 학습되는 변수 $x$와 정확히 희소성을 담당하는 변수 $z$를 분리하여 최적화한다. 그 결과로 제안된 방법이 SAFE이며, projection 자체를 일반화하여 magnitude 외의 saliency를 흡수하도록 확장한 버전이 SAFE+ 이다.

실험 결과는 SAFE가 실제로 더 sparse하고 더 flat한 해로 수렴하며, 이미지 분류와 LLM pruning 모두에서 강한 성능을 보인다는 점을 보여준다. 특히 label noise, common corruption, adversarial perturbation 환경에서도 성능 저하가 덜해, 단순한 compression 기법을 넘어 robust sparse optimization framework로 이해를 확장시키고자 한다.

---

### 1. Introduction

현대 딥러닝 모델은 대규모 데이터와 초과매개변수화된 네트워크 덕분에 모델의 표현력은 크게 증가했지만, 그만큼 실제 배포와 추론 단계에서의 과도한 계산량과 메모리 비용으로 인한 문제가 심각해졌다. 이러한 배경에서 pruning, quantization, distillation 같은 model compression 기법이 활발히 연구되어 왔고, 그중 pruning은 중복된 파라미터를 제거하여 효율성을 높이는 가장 직접적인 방법으로 자리 잡았다.

하지만 pruning의 가장 근본적인 한계는 높은 sparsity로 갈수록 성능 저하가 필연적이다. 기존 연구들은 중요도가 낮은 가중치를 heuristic하게 제거하거나, pruning 이후 retraining으로 성능을 복구하는 방향에 집중해 왔다. 이보다 더 근본적으로 접근해보자면, '왜 sparse model은 성능이 쉽게 무너지는가?'에 대한 논의를 단순히 parameter count 감소로만 설명하지 않고, sparse solution이 놓이는 손실 지형의 기하학적 성질에서 찾고자 하는 것이 논문의 논의거리이다.

이 문제의식은 flat minima 연구와 바로 연결된다. 잘 일반화되는 해는 sharp한 valley보다 넓고 완만한 valley에 놓이는 경향이 있으며, 이를 명시적으로 유도하는 대표적 방법이 SAM(Sharpness-Aware Minimization)이다. SAM은 현재 파라미터 한 점의 손실만 줄이는 것이 아니라, 그 주변 작은 neighborhood 전체에서 손실이 낮은 해를 찾도록 학습을 유도한다. 이에 대하여 pruning도 같은 관점으로 재해석할 수 있다고 본다. 즉, pruning이 잘 되려면 단순히 sparse한 해가 아니라 sparse하면서도 flat한 해가 필요하다는 것이다.

기존의 SAM-inspired pruning 연구들은 SAM으로 학습한 후 pruning이 잘 되길 기대하거나, compression에 덜 민감한 해를 찾는 방향에 머무는 경우가 많았다. 반면 이 논문은 pruning 자체를 sharpness-aware sparsity-constrained optimization 문제로 세우고, 이를 augmented Lagrangian 기반으로 명시적으로 푼다. 따라서 SAFE는 단순한 heuristic이 아니라, sparsity와 flatness를 하나의 목적 아래 공동으로 최적화하는 구조적 방법으로 접근하는 데에 의미가 있다.

---

### 2. Background

SAFE가 어떤 문제의식과 수학적 수식 접근에 대한 아이디어를 논하고자 한다. 한 축은 희소성 제약 최적화 이고, 다른 한 축은 flat minima를 선호하는 robust optimization 이다. 이후 Method 장에서 SAFE가 제안될 때, 이 두 축이 하나의 알고리즘으로 결합된다.

---

#### 2.1. Sparsity

가장 기본적인 sparse optimization 문제는 다음과 같이 쓸 수 있다.

$$
\min_{\||x\||_0 \le d} f(x)
$$

($f(x)$는 최소화하려는 목적함수, $\|x\|_0$는 0이 아닌 원소의 개수, $d$는 유지하려는 파라미터 수)
목표는 L-0 normㅔ 따라 non-zero 원소가 $d$개 이하인 해들 중 손실이 가장 낮은 해를 찾는 것이다. 문제는 $\ell_0$ 제약이 이산적이고 조합론적이어서, 정확한 최적해를 찾으려면 사실상 가능한 모든 mask 조합을 탐색해야 한다는 데 있다.

이 어려움 때문에 고전적으로는 여러 우회 전략이 사용되었다. LASSO는 $\ell_0$ 제약을 $\ell_1$ regularization으로 완화했고, FISTA나 iterative hard thresholding은 proximal 혹은 thresholding 기반으로 sparse solution을 효율적으로 찾으려 했다. 신경망 분야에서는 OBD와 OBS처럼 2차 정보를 활용해 특정 파라미터를 제거했을 때 손실 증가를 근사하는 방법도 등장했다.

딥러닝에서 sparsity는 적용 시점에 따라 크게 세 종류로 나뉜다. 학습 전 pruning은 sparse training 효율을 높이는 데 유리하고, 학습 중 pruning은 모델이 훈련되는 과정에서 원하는 sparse 구조로 유도할 수 있어 일반적으로 가장 좋은 성능을 내는 편이며, 학습 후 pruning은 이미 학습된 대형 모델을 낮은 비용으로 압축하는 데 적합하다. 특히 LLM에서는 전체 재학습이 거의 불가능하므로, block-wise reconstruction error minimization 같은 post-training pruning이 널리 쓰인다.

그럼에도 높은 sparsity 내 기존 dense model의 성능을 유지하는 일은 여전히 어렵다. 결국 많은 방법이 다양한 saliency score나 heuristic에 의존하게 되는데, 이 논문에선 희소성 자체를 제약 최적화 문제에 대해 그 안에 flatness까지 포함하는 접근을 시도한다.

---

#### 2.2. Flat Minima

flat minima란, 파라미터를 조금 움직여도 손실이 급격히 커지지 않는 넓고 완만한 영역을 의미한다. 딥러닝 최적화 연구는 잘 일반화되는 해가 종종 flat minima에 놓인다는 경험적 사실을 반복적으로 보여 왔다. 반대로 sharp minima는 아주 작은 perturbation에도 손실이 크게 증가하는 해이다. 이 관점은 mini-batch training의 일반화 성질, large-batch 학습의 일반화 gap, 그리고 모델 robustness와도 깊게 연결되어 있다.

이러한 연구에서 나오게 된 대표적 방법이 SAM이다.  

$$
\min_x \max_{\||\epsilon\||_2 \le \rho} f(x+\epsilon)
$$

수식의 의미는 현재 점 $x$ 하나의 손실만 줄이는 것이 아니라, 반경 $\rho$ 안에 있는 perturbation 전체를 고려했을 때도 손실이 낮은 해를 찾겠다는 것이다. 만약 어떤 해가 sharp하다면, 아주 작은 $\epsilon$만으로도 손실이 크게 증가하므로 inner maximization 값이 커지고, outer minimization은 그러한 해를 피하게 된다. 결과적으로 SAM은 자연스럽게 flat minima를 선호한다.

1차 Taylor approximation을 쓰면 inner maximization의 해는 gradient 방향으로 근사된다.

$$
\epsilon^\star(x) \approx \rho \frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$

따라서 실제 업데이트는 현재 파라미터에서 gradient 방향으로 약간 이동한 지점의 gradient를 계산하여 수행된다. 이 방식은 다양한 비전과 언어 과제에서 일반화와 robustness 향상에 효과적이라고 알려져 있다.

이 논문이 중요한 이유는 바로 이 sharpness-aware 관점을 pruning에 직접 접목했다는 데 있다. 즉, sparse model의 성능 저하를 줄이기 위해서는 단순히 어느 가중치를 남길지보다, 남겨진 sparse model이 어떤 손실 지형 위에 놓이는지가 중요하다고 본다.

---

### 3. Method

앞서 다룬 방법론적인 논의를 수식으로 검증해볼 수 있다. 앞선 작업으로부터 pruning을 크기가 작은 가중치를 제거하는 과정이 아니라, 희소성과 평탄성을 동시에 만족하는 sparse solution을 찾는 constrained robust optimization 문제로 재정의한다. 그리고 그 문제를 실제로 풀기 위해 augmented Lagrangian과 ADMM 구조를 사용한다.

---

#### 3.1. Problem Formulation

SAFE의 출발점은 다음 문제이다.

$$
\min_{\|x\|_0 \le d} \max_{\|\epsilon\|_2 \le \rho} f(x+\epsilon)
$$

($d$는 남길 파라미터 수이며, $\rho$는 flatness를 얼마나 강하게 요구할지 결정하는 반경)

바깥 minimization은 sparse constraint를 만족하는 파라미터를 찾는 과정이고, 안쪽 maximization은 그 주변의 가장 불리한 perturbation까지 고려하는 과정이다. 따라서 목적은 단순히 손실이 낮은 sparse model이 아니라, 작은 교란에도 손실이 급격히 증가하지 않는 sparse and flat solution을 찾는 것이다.

$\rho$가 커질수록 더 넓은 neighborhood에서 안정적인 해가 선호된다. 이 formulation은 pruning의 성능 저하를 단지 capacity 감소의 부산물이 아니라, geometry-aware optimization의 실패로 해석할 수 있게 만든다.

---

#### 3.2. Augmented Lagrangian Based Approach

하지만 위 문제는 개별적인 방식으로 직접 푸는 것은 불가능하다. $\ell_0$ 제약은 이산적이고, 신경망 손실은 강한 비선형성을 가지므로 순수한 Lagrangian duality나 projected gradient descent는 각각 한계를 가진다. Lagrangian만으로는 $\ell_0$ 제약이 다루기 어렵고, 단순 projection은 비선형 딥넷에서 학습을 지나치게 불안정하게 만들 수 있다. 이러한 단점을 해결하고 장점을 결합하기 위해 augmented Lagrangian 을 사용한다.

먼저 변수 분할을 도입하여, objective minimization을 담당하는 변수 $x$와 sparse constraint를 직접 만족하는 변수 $z$를 분리한다.

$$
\min_{x,z} \max_{\|\epsilon\|_2 \le \rho} f(x+\epsilon) + I_{\|\cdot\|_0 \le d}(z)
\quad \text{s.t. } x=z
$$

$$
I_{\|\cdot\|_0 \le d}(z)=
\begin{cases}
0 & \text{if } \|z\|_0 \le d \\
\infty & \text{otherwise}
\end{cases}
$$

(indicator function)

이후 penalty term을 더한 augmented Lagrangian을 구성하면, scaled dual variable $u$를 사용하여 다음과 같은 반복 구조를 얻는다.

$$
x^{k+1}=
\arg\min_x \max_{\|\epsilon\|_2 \le \rho} f(x+\epsilon)
+
\frac{\lambda}{2}\|x-z^k+u^k\|_2^2
$$

$$
z^{k+1}=
\operatorname{proj}_{\|\cdot\|_0 \le d}(x^{k+1}+u^k)
$$

$$
u^{k+1}=u^k + x^{k+1}-z^{k+1}
$$

$x$-step은 손실과 flatness를 고려해 연속적으로 최적화되는 단계이고, $z$-step은 현재 해를 정확히 sparse set 위로 projection하는 단계이며, $u$-step은 둘의 차이를 누적해 이후 반복에서 일치성을 강제한다. 결국 SAFE는 학습 가능한 dense-like 변수와 정확한 sparse proxy를 동시에 유지하면서, 둘을 점진적으로 일치시키는 구조로 이해할 수 있다.

---

#### 3.3. x-minimization

먼저 inner maximization은 SAM과 같은 방식으로 1차 근사를 이용해 푼다.

$$
\epsilon^\star(x)
\approx
\arg\max_{\|\epsilon\|_2 \le \rho} \left( f(x)+\epsilon^\top \nabla f(x) \right)=
\rho \frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$

이를 목적함수에 대입하면 $x$-step은 다음 문제로 바뀐다.

$$
x^{k+1}=
\arg\min_x
f(x+\epsilon^\star(x))
+
\frac{\lambda}{2}\|x-z^k+u^k\|_2^2
$$

이제 gradient를 계산하면, SAM과 마찬가지로 $\nabla \epsilon^\star(x)$에 의한 고차항을 무시하는 근사 아래 다음 식을 얻는다.

$$
\nabla_x
\left(
f(x+\epsilon^\star(x))
+
\frac{\lambda}{2}\|x-z^k+u^k\|_2^2
\right)=
\nabla f\!\left(x+\rho\frac{\nabla f(x)}{\|\nabla f(x)\|_2}\right)
+
\lambda(x-z^k+u^k)
$$

따라서 실제 업데이트는 다음과 같다.

$$
x_k^{(t+1)}=
x_k^{(t)}
-\eta^{(t)}
\left(
\nabla f\!\left(
x_k^{(t)}
+
\rho \frac{\nabla f(x_k^{(t)})}{\|\nabla f(x_k^{(t)})\|_2}
\right)
+
\lambda(x_k^{(t)}-z_k+u_k)
\right)
$$

첫 번째 항은 sharpness-aware gradient로 flat minima를 찾게 하고, 두 번째 항은 sparsity constraint에 가까워지도록 하는 penalty term이다. 즉, SAFE의 $x$-update는 flatness와 sparsity를 번갈아 강제하는 것이 아니라, 하나의 gradient update 안에서 동시에 반영 한다.

---

#### 3.4. Extension to Generalized Projection

기본 SAFE에서 $z$-step은 Euclidean distance 기준의 $\ell_0$ projection이며, 결국 magnitude가 큰 파라미터를 남기는 hard thresholding과 동일하다. 그러나 실제 pruning에서는 magnitude만으로 중요도를 판단하는 것이 항상 최선은 아니다. gradient sensitivity, Hessian curvature, activation statistics처럼 목적함수와 더 직접적으로 연결되는 saliency가 더 좋은 성능을 내는 경우가 많다.

이를 위해 SAFE+는 projection 자체의 거리 개념을 일반화한다. 양의 정부호 대각행렬 $P$를 도입하여 다음과 같이 정의한다.

$$
\operatorname{proj}_{\|\cdot\|_0 \le d}^{P}(v)
=
\arg\min_{\|z\|_0 \le d}
\frac{1}{2}\|z-v\|_P^2
=
\arg\min_{\|z\|_0 \le d}
\frac{1}{2}(z-v)^\top P(z-v)
$$

좌표별로 보면, 어떤 좌표를 0으로 보냈을 때의 비용은 대략 $\frac{1}{2}P_{ii}v_i^2$ 이다. 따라서 projection은 결국 $P_{ii}v_i^2$ 가 큰 좌표를 남기는 효과를 가지며, 이는 $\sqrt{P_{ii}}|v_i|$ 형태의 saliency score를 사용한 pruning으로 해석할 수 있다.

이 일반화는 여러 기존 pruning 기준을 하나의 틀 안에 넣는다.

##### IF $P=I$,

기본 SAFE와 동일하며, 단순 magnitude pruning과 대응된다.

##### IF $P=\operatorname{diag}(\nabla^2 f(x))$,

OBD류의 2차 pruning과 연결된다.

##### IF $P=\operatorname{diag}(\nabla f(x)\nabla f(x)^\top)$,

SNIP류의 1차 민감도 기반 pruning과 연결된다.

##### IF Wanda projection,

LLM pruning의 Wanda는 특정 layer activation $A$에 대해 $P=\operatorname{diag}(A^\top A)$ 로 해석할 수 있다.

즉 SAFE+의 의미는 새로운 pruning score 하나를 제안했다 것이 아닌, projection metric을 바꾸는 방식으로 다양한 saliency를 제약 최적화 내부에 통합했다는데에 있다. 이 점이 SAFE+를 단순한 heuristic 확장이 아니라, 보다 일반적인 sparse optimization framework로 만들어 준다.

---

#### 3.5. Final Algorithm: SAFE and SAFE+

최종 알고리즘은 크게 세 흐름으로 요약된다. 첫째, $x$는 sharpness-aware gradient를 통해 flat한 방향으로 학습된다. 둘째, 일정 간격 $K$마다 현재의 $x+u$를 sparse set에 projection하여 $z$를 갱신한다. 셋째, dual variable $u$를 업데이트하여 $x$와 $z$의 차이가 장기적으로 줄어들게 만든다.

실제로 논문 알고리즘은 다음과 같은 직관으로 읽을 수 있다. SAFE는 학습 도중 항상 현재 파라미터가 가장 가까운 sparse point에서 얼마나 떨어져 있는지를 관찰하고, 그 sparse point를 $z$에 기록한다. 그런 다음 $x$를 학습할 때는 단순히 training loss만 줄이는 것이 아니라, flat minima를 찾는 방향으로 움직이면서 동시에 sparse proxy 쪽으로도 조금씩 끌려가게 만든다. 이 덕분에 마지막에 갑자기 hard pruning을 가하는 방법보다 손실 폭증이 덜하다.

논문은 실제 비전 실험에서 penalty parameter $\lambda$를 0에서 목표값까지 cosine 형태로 증가시키는 스케줄링을 사용한다. 이는 초기 학습 단계에서는 표현 학습을 충분히 진행하고, 후반부로 갈수록 sparsity constraint를 강하게 반영하기 위한 설계이다. 이 선택은 Appendix의 ablation에서 실제로 성능 이점을 보인다.

---

#### 3.6. Convergence Analysis

이론 분석의 목적은 SAFE가 단순 heuristic이 아니라, well-founded optimization algorithm이라는 점을 보이는 데 있다. 논문은 $f$가 lower-bounded, $\beta$-smooth, $\mu$-weakly convex라는 표준 가정을 둔다.

먼저 다음 augmented objective를 정의한다.

$$
\hat{L}(x)=f(x)+\frac{\lambda}{2}\|x-z+u\|_2^2
$$

부록의 분석에 따르면, $f$가 $\beta$-smooth이고 $\mu$-weakly convex이면 $\hat{L}(x)$는 $(\beta+\lambda)$-smooth하고, $\lambda>\mu$일 때 $(\lambda-\mu)$-strongly convex가 된다. 즉, penalty term은 단순히 제약 위반을 벌점화하는 역할만 하는 것이 아니라, $x$-subproblem의 기하를 더 안정적인 방향으로 바꿔 준다.

SAFE의 $x$-update는 정확한 $\nabla \hat{L}(x)$ 대신, perturbation이 반영된 sharpness-aware gradient를 사용한다. 논문은 이 둘의 차이가 smoothness로 제어 가능하며, step size와 perturbation radius가 적절한 조건을 만족하면 결국 $\nabla \hat{L}(x^{(t)}) \to 0$ 임을 보인다. 다시 말해, $x$-update는 augmented Lagrangian에 대한 stationary point 쪽으로 수렴한다.

이후 기존 ADMM 수렴 결과를 결합하여, SAFE의 limit point가 원래 sparsity-constrained optimization 문제의 $\delta$-stationary point가 됨을 보인다. 이 결과는 SAFE가 sparse and flat solution을 찾는 방향으로 설계되었을 뿐 아니라, 적어도 sparse constrained optimization의 stationary point라는 엄밀한 의미에서 잘 정의된 알고리즘임을 보여준다.

---

### 4. Experiments

실험 장의 목적은 세 가지이다. 첫째, SAFE가 실제로 sparse하고 flat한 해를 만드는지 확인한다. 둘째, 이미지 분류와 LLM pruning에서 성능 향상이 있는지 검증한다. 셋째, noisy data와 corruption 환경에서도 robust한지 살핀다. 이 구성은 Method 장의 주장과 정확히 대응된다.

---

#### 4.1. Convergence to Sparse and Flat Solutions

가장 먼저 논문은 간단한 MLP와 MNIST를 이용해 SAFE가 실제로 어떤 해에 수렴하는지를 시각적으로 보여준다. Figure 1의 weight distribution을 보면, dense training은 가중치가 넓게 퍼져 있는 반면 SAFE는 0 근처에 강하게 집중되어 있어 sparse-friendly한 구조를 형성한다. 즉 SAFE는 학습 과정 자체가 자연스럽게 pruning 가능한 파라미터 구성을 만들도록 유도한다.

더 중요한 것은 손실 지형의 차이이다. 논문은 ADMM과 SAFE로 찾은 해의 loss landscape를 시각화하고, 최대 Hessian 고유값을 sharpness 지표로 비교한다. 결과적으로 SAFE의 sharpness는 0.09, ADMM은 0.2로 보고되며, SAFE가 더 넓고 완만한 valley에 놓인다는 점이 분명히 드러난다. 이 결과는 SAFE가 단지 sparsity만 강제하는 것이 아니라, flatness까지 함께 유도하는 알고리즘 이라는 논문의 핵심 주장을 직접적으로 뒷받침한다.

---

#### 4.2. Evaluations on Image Classification

이미지 분류 실험은 CIFAR-10/100 데이터셋에서 VGG-19, ResNet-20, ResNet-32를 다양한 sparsity 수준으로 pruning하면서 SAFE를 PBW, GMP, LTH, ADMM, MLPrune과 비교하는 방식으로 진행된다. 중요한 점은 논문이 추가 retraining 없이 batch-norm tuning만 수행하고 결과를 비교했다는 것이다. 즉, pruning 이후의 성능 복구를 긴 재학습에 의존하지 않는 비교적 엄격한 설정이다.

결과는 전반적으로 SAFE가 거의 모든 구간에서 가장 강하거나 가장 안정적인 성능을 보인다는 점을 보여준다. 특히 extreme sparsity로 갈수록 차이가 커진다. 예를 들어 VGG-19/CIFAR-10에서 99.5% sparsity일 때 SAFE는 93.56%를 기록하는 반면, ADMM은 88.53%, GMP는 90.63%이다. ResNet-32/CIFAR-100의 99.5% sparsity에서는 SAFE가 51.45%인데 ADMM은 12.34%까지 무너진다. 이 결과는 SAFE가 높은 sparsity에서도 급격한 성능 붕괴를 늦추는 방향으로 학습을 이끈다 는 해석과 잘 맞는다.

핵심은 SAFE가 좋은 pruning mask를 사후적으로 찾는 것이 아니라, 처음부터 pruning 이후에도 버틸 수 있는 형태의 representation을 학습 과정에서 만든다 는 점이다. 따라서 sparse projection 이후의 loss jump가 작고, 구조적으로 더 강인한 sparse minima를 얻게 된다.

---

#### 4.3. Evaluation on Large Language Model Pruning

LLM 실험은 SAFE의 일반성을 보여주는 파트이다. 논문은 LLaMA-2 7B/13B와 LLaMA-3 8B를 대상으로 50%, 60%, structured 4:8, 2:4 sparsity를 적용한다. 여기서는 비전 실험처럼 전체 training 과정에서 pruning을 수행하는 대신, 각 transformer block에 대해 reconstruction error minimization(REM) objective를 순차적으로 푸는 post-training pruning 프레임을 사용한다. SAFE는 이 block-wise objective 위에 적응되며, SAFE+는 $z$-step에 Wanda projection을 사용해 더 강한 saliency 정보를 반영한다.

결과적으로 기본 SAFE도 state-of-the-art LLM pruning 기법과 충분히 경쟁력 있는 perplexity를 보이며, SAFE+는 모든 모델과 sparsity 설정에서 가장 좋은 결과를 내는 경우가 많다. 예를 들어 LLaMA-2 13B를 60% sparsity로 pruning할 때 SAFE+는 WikiText/C4에서 6.78 / 9.02를 기록하며, SparseGPT의 8.31 / 10.85와 ALPS의 7.54 / 9.87보다 우수하다. LLaMA-3 8B의 50% sparsity에서도 SAFE+는 8.62 / 13.26으로, SparseGPT와 Wanda보다 낮은 perplexity를 보인다.

이 결과는 두 가지를 의미한다. 첫째, SAFE의 핵심 원리는 비전 모델에만 한정된 trick이 아니라, 대규모 언어모델의 post-training pruning에도 적용될 수 있는 일반적 최적화 관점이라는 점이다. 둘째, generalized projection을 도입한 SAFE+가 실제로 큰 효과를 발휘한다는 점이다. 또한 논문은 SAFE가 ALPS보다 runtime 측면에서도 더 효율적이며, ALPS가 SAFE보다 약 2.54배 더 오래 걸린다고 보고한다.

---

#### 4.4. Robustness to Noisy Data

이 절은 SAFE가 단순히 clean accuracy만 높은 것이 아니라, 현실적인 데이터 노이즈에도 강하다는 점을 보여준다. 논문은 세 가지 조건을 본다. 학습 시 label noise, 추론 시 common corruption, 그리고 adversarial perturbation 이다.

먼저 label noise 실험에서는 CIFAR-10의 라벨을 25%, 50%, 75% 비율로 무작위 오염시키고, ResNet-20을 ADMM과 SAFE로 각각 학습한다. 결과는 매우 강력하다. 예를 들어 80% sparsity와 50% label noise에서 ADMM은 62.67%인 반면 SAFE는 86.55%를 기록한다. 95% sparsity와 75% noise에서도 ADMM은 39.68%, SAFE는 64.25%이다. 전반적으로 SAFE는 모든 noise ratio와 sparsity 구간에서 10~30%p 수준의 큰 이득을 보인다.

논문이 흥미롭게 지적하는 부분은 ADMM이 label noise를 완화하기 위해 sparsity 자체에 과도하게 의존하는 경향을 보인다는 점이다. 25% noise에서는 sparse double descent 유사 패턴까지 나타난다. 반면 SAFE에서는 이런 불안정성이 거의 보이지 않는다. 이는 sharpness minimization이 일종의 regularizer처럼 작동해 noisy label에 대한 과적합을 줄였다고 해석할 수 있다.

추론 시 corruption에서도 SAFE의 이점은 유지된다. CIFAR-10C의 common corruption 평균 정확도는 90% sparsity에서 ADMM 70.06, SAFE 73.98이며, $l_\infty$-PGD에서는 ADMM 49.81, SAFE 56.43이다. 99% sparsity처럼 더 극단적인 조건에서도 SAFE는 common corruption과 adversarial setting 모두에서 더 강한 성능을 보인다. 즉 SAFE의 flatness 유도는 일반화 성능 향상뿐 아니라 노이즈와 공격에 대한 회복력 향상 으로도 연결된다.

---

#### 4.5. Comparison with Other SAM-based pruner

이 절의 목적은 SAFE를 단순한 “SAM을 pruning에 얹은 방법”과 구분하는 데 있다. 비교 대상은 IMP+SAM과 CrAM이다. IMP+SAM은 iterative magnitude pruning 과정에 SAM을 적용하는 방식이고, CrAM은 compression-aware objective를 통해 compression 이후 손실 증가를 줄이려는 방식이다. 또한 논문은 CrAM+처럼 원래 gradient를 추가로 더하는 변형과, 이에 대응하는 SAFE+SG도 비교한다.

결과를 보면, 기본 SAFE는 IMP+SAM과 CrAM보다 일관되게 안정적이며, SAFE+SG는 중간 sparsity에서는 CrAM+와 비슷하거나 약간 더 좋고, extreme sparsity에서는 더 우수하다. 예를 들어 ResNet-20/CIFAR-10에서 99.5% sparsity일 때 IMP+SAM(cubic)은 73.73%, CrAM+는 81.30%, SAFE는 79.55%, SAFE+SG는 85.85%이다. 특히 SAFE+SG가 가장 높은 성능을 보인다는 점은, SAFE의 구조 위에 보조 gradient를 더했을 때도 효과가 확장될 수 있음을 보여준다.

그러나 이 절의 더 중요한 메시지는 수치 자체보다 해석에 있다. CrAM은 auxiliary trick이 없는 기본 형태로는 매우 불안정하며, 성능 향상이 CrAM+ 같은 추가 기법에 크게 의존한다. 반면 SAFE는 그런 추가 장치 없이도 이미 강한 성능을 보인다. 이는 SAFE의 장점이 단지 SAM-style perturbation 때문이 아니라, augmented Lagrangian의 smooth penalization과 split-variable 구조 자체에 있다 는 논문의 주장과 맞닿아 있다.

---

### 5. Conclusion

이 논문은 pruning을 희소성 문제로만 보지 않고, 희소성과 평탄성을 동시에 고려하는 constrained optimization 문제 로 재정의했다는 점에서 의미가 크다. SAFE는 augmented Lagrangian과 ADMM 기반의 split-variable 구조를 통해, 학습 가능한 변수 $x$는 flat minima를 향해 움직이게 하고, sparse proxy $z$는 정확한 sparsity constraint를 담당하게 만든다. 이 설계 덕분에 pruning은 더 이상 마지막에 갑자기 hard thresholding을 가하는 절차가 아니라, 학습 전반에 걸쳐 sparse하고 flat한 해를 점진적으로 형성하는 과정 이 된다.

실험적으로도 SAFE는 이미지 분류와 LLM pruning 모두에서 강력한 결과를 보이며, 특히 extreme sparsity와 noisy environment에서 더 큰 이점을 드러낸다. SAFE+는 generalized projection을 통해 다양한 saliency score를 같은 프레임 안에서 해석하고 활용할 수 있게 하며, 실제로 LLM pruning에서는 기존 강력한 baselines를 넘어서는 성능을 보인다.

결국 이 논문이 남기는 핵심 메시지는 분명하다. 좋은 sparse model은 단지 sparse하기만 해서는 안 되고, 반드시 flat해야 한다. SAFE는 바로 그 원리를 이론과 알고리즘, 그리고 실험으로 연결한 작업이다.

---

### Acknowledgements

논문은 본 연구가 POSTECH 관련 인공지능 대학원 프로그램, 인과추론 기반 vision-language 의사결정 연구 과제, 그리고 한국연구재단 지원을 포함한 복수의 연구비 지원을 받았음을 밝힌다. 이는 본 연구가 이론적 기여뿐 아니라 실제 대규모 실험을 수행할 수 있는 연구 인프라 위에서 진행되었음을 보여준다.

---

### Impact Statement

저자들은 이 연구가 기계학습의 이론적 이해와 실제 응용 모두에 영향을 줄 수 있다고 본다. 논문 자체는 즉각적인 사회적 위해를 직접적으로 강조하지 않지만, 대규모 모델을 더 효율적으로 만들고 real-world noise에 더 강인하게 만드는 기술은 향후 다양한 응용 분야에 영향을 줄 수 있으므로, 그 파급효과에 대한 지속적인 논의가 필요하다는 입장을 취한다.

---

### A. Convergence analysis of SAFE

부록 A는 본문의 이론적 주장을 엄밀하게 정리하는 역할을 한다. 핵심 목표는 SAFE의 $x$-update가 augmented Lagrangian의 stationary point로 수렴하고, 이를 바탕으로 전체 알고리즘이 sparsity-constrained optimization의 $\delta$-stationary point와 연결됨을 보이는 것이다.

우선 다음 augmented objective를 둔다.

$$
\hat{L}(x)=f(x)+\frac{\lambda}{2}\|x-z+u\|_2^2
$$

이 함수는 penalty term 덕분에 원래 목적함수보다 더 좋은 기하학적 성질을 가진다. 부록은 $f$가 $\beta$-smooth, $\mu$-weakly convex일 때 $\hat{L}$이 $(\beta+\lambda)$-smooth하고, $\lambda>\mu$이면 $(\lambda-\mu)$-strongly convex함을 먼저 보인다. 이 성질이 뒤의 수렴 증명의 기반이 된다.

---

#### A.1. Proof of Theorem 3.5

이 절의 핵심은 SAFE가 사용하는 sharpness-aware gradient가 정확한 $\nabla \hat{L}(x)$와 얼마나 다른지를 통제하는 것이다. 논문은 $x$-update에 쓰이는 gradient를

$$
g^{(t)}
=
\nabla f\!\left(
x^{(t)} + \rho^{(t)} \frac{\nabla f(x^{(t)})}{\|\nabla f(x^{(t)})\|}
\right)
+
\lambda(x^{(t)}-z+u)
$$

로 두고, smoothness를 이용해

$$
\|g^{(t)}-\nabla \hat{L}(x^{(t)})\| \le \beta \rho^{(t)}
$$

와 같은 형태의 상계를 얻는다. 즉 perturbation으로 인한 bias는 $\rho^{(t)}$에 비례하여 제어된다.

이후 descent lemma와 보조수열을 도입해, 적절한 step size와 perturbation radius 조건

$$
\sum_{t=1}^{\infty}\eta^{(t)}=\infty, \qquad
\sum_{t=1}^{\infty}\eta^{(t)}\rho^{(t)}<\infty, \qquad
\limsup_t \rho^{(t)} < \frac{1}{\beta}
$$

아래에서

$$
\nabla \hat{L}(x^{(t)}) \to 0
$$

임을 증명한다. 즉, SAFE의 $x$-minimization은 단순한 heuristic update가 아니라 augmented objective에 대한 stationary point를 향하는 반복으로 정당화된다.

---

#### A.2. Proof of Theorem 3.6

앞 절에서 $x$-update의 수렴을 확보했으므로, 여기서는 이를 classical ADMM 이론과 결합한다. 논문은 SAFE의 각 외부 반복에서 $x$-subproblem이 충분히 잘 풀렸을 때, 전체 반복이 표준 ADMM 분석이 요구하는 형태를 만족함을 사용한다. 그 결과 limit point $(\bar{x}, \bar{z}, \bar{u})$에 대해 $\bar{x}$가 원래 sparsity-constrained optimization 문제의 $\delta$-stationary point가 됨을 보인다.

핵심 의미는 SAFE가 단지 empirical trick이 아니라, 제약 최적화 이론 위에 놓인 pruning 알고리즘 이라는 점이다. 이론적 기여 자체가 매우 복잡한 것은 아니지만, 최근 pruning 연구에서 자주 볼 수 있는 ad-hoc 설계와 달리 SAFE는 비교적 명확한 수학적 기반을 갖는다.

---

### B. Experimental Details

부록 B는 실험 재현에 필요한 설정을 정리한다. 모든 결과는 3개의 서로 다른 seed에서 실행되며 평균과 표준오차로 보고된다. 또한 SAFE 고유의 하이퍼파라미터인 perturbation radius, dual-update interval, penalty parameter가 어떤 범위에서 탐색되었는지도 함께 제공된다.

---

#### B.1. Hyperparameters

비전 실험에서는 주로 SGD, cosine learning rate schedule, batch size 128, weight decay 0.0001, momentum 0.9가 사용된다. ResNet-20은 200 epoch, 그 외 비전 모델은 300 epoch로 학습한다. 반면 언어모델 pruning에서는 Adam, linear learning rate schedule, batch size 8, 30 epoch, 2 epoch warm-up을 사용한다.

SAFE 고유 하이퍼파라미터를 보면, 비전에서는 $\rho \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$, 언어에서는 더 작은 범위를 사용한다. Dual-update interval $K$는 비전에서 넓은 범위를 탐색하고, 언어에서는 $\{16, 32, 64\}$ 수준으로 제한한다. 중요한 점은 이 값들이 모든 task마다 완전히 새로 튜닝된 것이 아니라, 대표 설정에서 선택된 뒤 대부분의 실험에 공통 적용되었다는 것이다. 이는 SAFE가 task-specific tuning에만 의존하는 방법이 아님을 시사한다.

---

#### B.2. Experimental Details in Section 4.1

Section 4.1의 toy experiment는 MNIST 위의 3-layer MLP로 수행된다. hidden dimension은 300, 100이며, sparsity 실험에서는 dense training과 SAFE를 90% sparsity 조건에서 비교한다. flatness 비교에서는 SAFE와 ADMM을 동일 sparsity에서 실행한 뒤, loss landscape를 시각화하고 power iteration으로 최대 Hessian 고유값을 추정한다.

이 구성은 Figure 1의 메시지를 뚜렷하게 전달하기 위한 것이다. 즉, SAFE가 실제로 더 sparse한 weight distribution을 만들고, 동시에 더 낮은 sharpness를 갖는 해를 찾는다는 점을 가장 간단한 환경에서 먼저 확인한다.

---

#### B.3. Experimental Details in Section 4.2

이미지 분류 실험에서는 batch-norm이 포함된 VGG와 standard ResNet 계열을 사용하며, random crop과 horizontal flip 같은 기본적인 data augmentation을 적용한다. SAFE와 대부분의 baseline 모두 같은 base optimizer인 SGD를 사용해 비교의 공정성을 확보한다. 실험은 단일 GPU 혹은 3개의 RTX 3090에서 수행된다.

또한 pruning 이후 긴 retraining 대신, 소수의 forward pass만으로 batch statistics를 다시 맞추는 batch-norm tuning(BNT)을 사용한다. 이는 계산 비용을 크게 늘리지 않으면서도 sparse projection 이후 발생하는 batch statistics mismatch를 일부 완화하기 위한 선택이다.

---

#### B.4. Experimental Details in Section 4.3

LLM pruning 실험에서는 SparseGPT의 설정을 따라 C4 데이터셋의 첫 shard에서 sequence length 2048의 샘플 128개를 무작위로 뽑아 calibration에 사용한다. 모델은 HuggingFace 허브의 LLaMA-2-7B, LLaMA-2-13B, LLaMA-3.1-8B를 사용하며, 단일 NVIDIA A6000/L40S GPU 혹은 Intel Gaudi2 환경에서 수행된다.

SAFE와 SAFE+는 30 epoch 동안 Adam으로 최적화되며, $(\beta_1,\beta_2)=(0.9, 0.95)$, weight decay는 사용하지 않는다. 이 실험은 full retraining이 아니라 block-wise REM objective를 반복적으로 푸는 post-training pruning 설정이라는 점에서, 대규모 언어모델 환경에 맞춘 현실적인 구성이다.

---

#### B.5. Implementation and Reproduction Details

논문은 재현 코드를 JAX와 PyTorch 두 프레임워크로 제공한다. 이미지 분류용 SAFE 구현은 JAX 기반이고, LLM pruning은 공식 구현과 pretrained checkpoint 지원이 풍부한 PyTorch 기반으로 작성되었다. Baseline 중 ADMM, GMP, Magnitude는 저자들이 직접 구현했고, SparseGPT, Wanda, ALPS는 공식 구현을 사용한다.

또한 LTH, PBW, MLPrune 일부 결과는 기존 논문의 보고 수치를 참조한다. 저자들은 비교의 공정성을 위해 아키텍처, 학습 epoch, 데이터 처리 방식 등을 가능한 한 선행연구와 맞추었다고 설명한다.

---

### C. Detailed Results for Image Classification Tasks

이 부록은 Figure 2를 수치로 상세하게 제시한다. 곡선만 볼 때는 전체 경향만 파악할 수 있지만, Table 7을 보면 SAFE의 우세가 어떤 조건에서 특히 큰지를 더 정확히 읽을 수 있다.

예를 들어 CIFAR-10의 VGG-19에서는 90%부터 99.5%까지 모든 sparsity에서 SAFE가 가장 높은 정확도를 기록한다. CIFAR-100의 ResNet-32에서는 99% sparsity에서 SAFE가 62.77%, ADMM이 49.13%, GMP가 58.10%이고, 99.5%에서는 SAFE가 51.45%, ADMM은 12.34%까지 떨어진다. 이 수치는 SAFE가 극단적인 sparsity 영역에서 특히 강하다 는 본문의 주장을 강하게 뒷받침한다.

즉, 부록 C는 SAFE가 평균적으로 조금 더 좋은 수준이 아니라, 성능 붕괴가 시작되는 어려운 구간에서 훨씬 더 늦게 무너지는 방법 임을 정량적으로 보여준다.

---

### D. Additional Comparison with SAM-based pruners

부록 D는 Section 4.5의 비교를 확장해, CrAM의 여러 변형과 LLM setting에서의 IMP+SAM까지 다룬다. 이 부록의 목적은 SAFE가 SAM-inspired pruning 계열 안에서 어디에 위치하는지를 더 명확히 보여주는 것이다.

---

#### D.1. Other variants of CrAM

저자들은 CrAM과 CrAM+ 외에도, iteration마다 target sparsity를 바꾸는 CrAMMulti와 CrAM+Multi를 비교한다. 또한 공정한 대응을 위해 SAFE에도 유사한 변형을 도입해 SAFEMulti, SAFE+SG,Multi를 함께 제시한다.

결과적으로 auxiliary trick을 넣으면 CrAM 계열의 성능이 개선되기는 하지만, SAFE 역시 비슷한 전략을 도입하면 함께 개선된다. 더 중요한 점은 기본 SAFE가 여전히 상당히 강한 baseline이라는 것이다. 즉, CrAM의 성능 향상을 단순히 robust optimization formulation 자체의 효과로만 보기 어렵고, projected point gradient 같은 추가 장치의 영향도 크다는 해석이 가능하다. SAFE는 이런 추가 장치 없이도 안정적인 성능을 보여 주기 때문에, 기본 구조 자체의 완성도가 더 높다고 볼 수 있다.

---

#### D.2. IMP+SAM on Language model pruning

이 절은 IMP+SAM을 LLaMA2-7B의 50% sparsity pruning에 적용해 SAFE와 비교한다. 결과는 매우 인상적이다. IMP+SAM은 C4 / WikiText perplexity가 18.27 / 176.00인 반면, SAFE는 8.91 / 6.79를 기록한다. 즉, image classification에서 보인 경향이 LLM pruning에서도 그대로 반복된다.

이 비교는 중요한 시사점을 준다. 단순히 SAM을 pruning 과정에 넣는 것만으로는 충분하지 않으며, sparsity와 sharpness를 어떤 최적화 구조 안에서 묶어 주느냐 가 성능을 좌우한다는 것이다. SAFE가 높은 이유는 바로 이 구조적 결합에 있다.

---

### E. Computation Cost Analysis LLM Pruning

부록 E는 LLM pruning에서 SAFE의 계산비용을 이론적·실험적으로 분석한다. pruning 품질이 좋더라도 계산량이 지나치게 크면 실제 활용이 어렵기 때문에, 이 파트는 방법의 실용성을 평가하는 데 중요하다.

---

#### E.1. Theoretical Time Complexity

논문은 single transformer block pruning 기준으로 SAFE, SparseGPT, Wanda, ALPS의 시간복잡도를 비교한다. SAFE의 복잡도는 대략

$$
O(L_B b k d^2)
$$

로 제시되며, 여기서 $L_B$는 block 내 layer 수, $b$는 batch size, $k$는 iteration 수, $d$는 hidden dimension이다. 즉 SAFE는 iteration 수에 따라 비용이 증가하지만, hidden dimension에 대해서는 quadratic scaling을 가진다.

반면 SparseGPT와 ALPS는 Hessian inverse나 eigendecomposition 때문에 cubic term이 포함된다. 따라서 대규모 hidden dimension을 갖는 LLM에서는 SAFE가 optimization-based method 중 비교적 유리한 scaling을 가진다고 볼 수 있다. 물론 Wanda처럼 one-shot 성격의 매우 가벼운 방법보다는 느리지만, 고차 선형대수 연산을 반복적으로 수행하는 방법들보다는 구조적으로 부담이 덜하다.

---

#### E.2. Wall-clock Time

실제 wall-clock 결과에서도 이러한 경향이 나타난다. LLaMA-2-7B의 첫 transformer block을 50% sparsity로 pruning할 때, Magnitude는 0.48초, Wanda는 3.98초, SparseGPT는 15.82초, SAFE는 310.68초, ALPS는 788.66초가 걸린다.

이 결과는 SAFE의 위치를 분명히 해 준다. SAFE는 one-shot heuristic보다 훨씬 무겁지만, optimization-based ADMM 계열인 ALPS보다 상당히 빠르다. 따라서 SAFE는 “가장 빠른 pruning 방법”이라기보다, 성능·강건성·계산비용 사이에서 균형을 잡은 방법 으로 이해하는 것이 맞다.

---

### F. Ablation Study

부록 F는 SAFE가 왜 작동하는지, 그리고 어떤 하이퍼파라미터가 어떤 역할을 하는지를 구체적으로 보여준다. 특히 penalty parameter $\lambda$, batch-norm tuning, $\lambda$ scheduling, dual-update interval $K$가 성능에 미치는 영향을 체계적으로 분석한다.

---

#### F.1. Effects of Penalty Parameter λ

Figure 3은 $\lambda$가 dense model 성능, sparse projection 이후 성능, 그리고 sparsity constraint까지의 거리에 어떤 영향을 미치는지를 보여준다. 전반적으로 $\lambda$가 커질수록 학습 중 파라미터가 sparse constraint에 더 가까워지므로, 마지막 projection 이후의 성능 하락은 줄어든다. 그러나 동시에 원래 dense model의 validation accuracy는 떨어진다.

이는 $\lambda$가 objective minimization과 constraint satisfaction 사이의 균형추 역할을 한다는 뜻이다. 작은 $\lambda$는 학습을 자유롭게 하지만 마지막 projection이 너무 급격해지고, 큰 $\lambda$는 projection 손실은 줄이지만 학습 자체가 지나치게 제약된다. 특히 target sparsity가 높아질수록 이 trade-off가 훨씬 더 민감해지며, 극단적 sparsity일수록 적절한 $\lambda$ 선택이 중요해진다.

---

#### F.2. Effects of Batch-norm Tuning

BNT는 최종 projection 이후 batch statistics를 다시 맞추는 절차이다. 논문은 이 기법이 특히 작은 $\lambda$에서 유효하다고 보고한다. 이유는 명확하다. $\lambda$가 작으면 학습 중 파라미터가 sparse constraint에서 멀리 떨어져 있고, 마지막 projection에서 큰 변화가 생기므로 batch-norm statistics mismatch가 커진다.

반면 높은 sparsity에서는 BNT만으로 성능을 충분히 회복하지 못하는 경우가 많다. 이는 이 구간의 성능 손실이 단순히 batch statistics 불일치 때문만이 아니라, sparse network 자체의 표현력 저하와 구조적 손실에서 더 크게 기인한다는 점을 시사한다.

---

#### F.3. Effect of λ Scheduling

논문은 constant penalty 대신, $\lambda$를 0에서 목표값까지 천천히 올리는 schedule이 성능에 어떤 영향을 주는지 실험한다. 결과적으로 linear와 cosine schedule은 모두 constant보다 좋으며, 특히 95% sparsity에서 이득이 크다. Table 12에 따르면 ResNet-20/CIFAR-10의 95% sparsity에서 constant는 90.78%, linear는 92.20%, cosine은 92.59%를 기록한다.

Figure 4는 그 이유를 설명한다. constant penalty는 초반부터 모델을 sparsity constraint 가까이 강하게 밀어 넣지만, scheduling은 초기에는 constraint에서 잠시 멀어질 자유를 허용한다. 이는 초기 학습 단계에서 충분한 표현 학습을 먼저 수행한 뒤, 후반부에 sparsity를 강하게 반영하는 것이 더 유리함을 의미한다. 다시 말해, 좋은 sparse model을 얻으려면 처음부터 강하게 prune하는 것이 아니라 학습 초반의 자유와 후반의 제약을 적절히 배치하는 과정 이 중요하다.

---

#### F.4. Effects of dual-update interval K

Dual-update interval $K$는 $z$와 $u$를 얼마나 자주 갱신할지 결정한다. Figure 5에 따르면 실험 범위 내에서는 대부분의 sparsity에서 $K$가 성능에 큰 영향을 주지 않는다. 이는 SAFE가 비교적 넓은 $K$ 범위에서 안정적으로 동작한다는 뜻이다.

그러나 95% sparsity처럼 더 어려운 조건에서는 $K$가 너무 크면 sparse constraint 방향으로 충분히 자주 끌어주지 못해, 최종 sparse model의 성능이 떨어진다. 즉 $K$가 지나치게 크면 dual ascent의 빈도가 줄어 constraint satisfaction이 약해지고, 결국 projection 시 손실이 커진다. 이 결과는 SAFE가 단순히 한 번 sparse mask를 정해 놓고 학습하는 방식이 아니라, 지속적인 dual interaction을 통해 sparsity와 학습 dynamics를 맞춰 가는 알고리즘 임을 다시 보여준다.

---  
  
Review by 변정우, Aerospace Engineering Undergraduate Researcher  
[Update - Time Log]  
* 2026.05.03: [Draft] 전체적인 내용 리딩 완료 및 초안 작성  
* 2026.05.04: [ver_1] part 1 수식 및 관련 내용 업데이트
* 2026.05.06: [ver_2] part 2,3,4 수식 및 관련 내용 업데이트
* 2026.05.0: [ver_1] part 1 수식 및 관련 내용 업데이트
