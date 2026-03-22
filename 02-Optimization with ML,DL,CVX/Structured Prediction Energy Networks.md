## Structured Prediction Energy Networks

### 0. 논문 정보 (Reference)
* Title: Structured Prediction Energy Networks
* Authors: David Belanger, Bishan Yang, Andrew McCallum
* Conference: International Conference on Machine Learning (ICML), 2017
* arXiv: 1511.06350
* OpenReview: https://openreview.net/forum?id=ryB37X-Cb

### Abstract

이 논문은 Structured Prediction Energy Networks(SPENs)에 관해 다룬다. 이는 구조화된 출력 $y$ 자체에 대해 딥러닝 기반의 에너지 함수 $E_{X}(y)$를 정의하고, 예측 시에는 네트워크 파라미터가 아니라 출력 변수 $y$에 대해 역전파를 수행하여 에너지를 반복적으로 최소화하는 것이다. 이 방식은 신경망처럼 $x$ ↦ $y$를 한 번에 통과시키는 것이 아니라, 출력 구조의 상호작용 자체를 딥 아키텍처로 학습할 수 있다는 점이 차별적이다.

특히 다중 라벨 분류를 대표 예시로, 라벨 수가 커질 때도 파라미터 수와 예측 1회당 복잡도가 선형적으로 증가하는 구조를 제시한다. 이 과정에서 라벨 사이의 상호작용을 고차(high-order)로 담을 수 있다. 그래프 모델로 표현하면 treewidth가 너무 커져 추론이 불가능해질 때도, SPEN에서는 신경망 안의 비선형 함수로 처리할 수 있다. 또한, 기존 CRF류 접근은 레이블 수 $L$이 커지면 파라미터 수와 추론 비용이 최소 $O(L^2)$ 이상으로 커지는 반면, SPEN은 설계를 잘 하면 표현력은 유지하면서도 복잡도를 $O(L)$ 수준으로 유지할 수 있다.  

---

### 1. Introduction

일반적인 구조화 예측은 입력 $x$에 대해 출력 $y$가 단일 클래스가 아니라 서로 의존적인 여러 변수의 집합일 때 활용된다. 이때 $y$의 개수는 출력 변수 수에 대해 지수적으로 증가하므로 좋은 모델은 단순히 정확한 점수 함수를 학습하는 것만으로는 충분하지 않다. 즉, 이 경우엔 표현력 있는 모델링과 효율적인 탐색 가능성이 동시에 필요하다.

이때 두 관점에서 구조화 예측을 생각해볼 수 있다.
- Feed-forward 관점: $y = f(x)$로 직접 예측한다.  
- Energy-based 관점: $$y = \arg\min_{y'} E_x(y')$$ 에너지를 최소화하는 출력을 찾는다.  

Feed-forward 방식은 입력을 바로 출력으로 보내는 방식이라 end-to-end 학습이 쉽고 빠르다. 반면 energy-based 방식은 예측 자체가 최적화 문제이기 때문에 복잡하지만, 출력 구조에 대한 도메인 지식을 직접 반영하기 쉽고 더 압축적인 파라미터화를 통해 적은 데이터에서도 강한 일반화가 가능할 수 있다.  

기존의 딥 구조화 예측은 CRF 같은 기존 그래프 구조를 먼저 정하고 그 그래프의 potential이 입력 $x$에 의존하는 방식을 딥 네트워크로 표현해 왔다. 즉, 딥러닝이 입력 표현은 학습해도 출력 구조 자체의 표현 학습과 structure learning에는 제약이 있다.  

이에 따라 SPEN은 에너지 함수 $E_x(\bar y)$를 딥 네트워크로 정의하고, 예측은 $\bar y$에 대한 gradient descent로 수행한다. 이렇게 하면 practitioners는 특정 그래프 구조에 얽매이지 않고 고차 상호작용까지 포함하는 훨씬 유연한 에너지 함수를 설계할 수 있다. 대신 예측은 전역 최적화가 아니라 비선형 비볼록 목적함수의 지역 최적화에 의존한다. 따라서 SPEN은 일종의 trade-off를 감수하며 다중 라벨 분류를 다루고자 한다.

---

### 2. Structured Prediction Energy Networks

먼저 수식적인 접근으로 구조화 예측을 다음과 같은 조합 최적화 문제로 둔다.

$$\min_{y} E_x(y)  
\quad\text{subject to } y \in \{0,1\}^{L}.$$  

이는 binary CRF를 포함한 다양한 구조화 예측 문제를 포괄한다. 다만 이 문제를 그대로 풀면 제약 집합이 꽤나 이산적이므로 다음과 같은 조건으로 연속 완화하여 문제를 다룬다.

$$\min_{\bar y} E_x(\bar y)  
\quad \text{subject to } \bar y \in [0,1]^L$$  

이때 $E_x(\bar y)$가 일반적으로 non-convex다. 따라서 이를 정확히 푸는 대신, SPEN은 $\bar y$에 대해 gradient descent를 적용해 근사적인 local minimum을 찾는다. 이때 제약집합 $[0,1]^L$ 안에서 최적화를 수행해야 하므로, projected gradient descent나 entropic mirror descent를 활용해야 한다.

이 논문에서는 후자를 선택한다. 각 반복점이 항상 $(0,1)^L$ 내부에 머무르므로 경계에서 발산하는 에너지나 loss를 더 안정적으로 다룰 수 있기 때문이다. 다만 이 방식이 항상 거의 0-1에 가까운 해를 돌려준다는 보장은 없다. 따라서 상황에 따라 마지막에 thresholding이나 rounding을 적용하거나 soft prediction을 그대로 유지할 수 있다.  

또한, $\bar y$를 도입했다고 해서 SPEN이 mean-field variational inference의 marginal probability와 같은 의미를 가진다고 할 순 없다. mean-field는 본래 확률 모델에서 정확한 추론이 어려울 때 variational objective를 유도하는 방식인 반면, SPEN은 그런 확률적 가정을 세우지 않는다. 이들은 분포를 근사하는 것이 아니라 예측 절차 자체가 최소화할 목적함수를 discriminative하게 학습한다.  

SPEN의 파라미터화는 Feature network $F(x)$(입력 $x$를 표현 벡터로 변환)와 Energy network $E(F(x), \bar y)$(입력 표현과 출력 후보를 받아 scalar energy를 반환)으로 진행한다. 즉, $$E_x(\bar y) = E(F(x), \bar y)$$이고 예측 시에는 $F(x)$를 먼저 한 번만 계산해 둔 뒤 $y$에 대한 gradient만 계산하면 된다. 이로 인해 iterative prediction에서 입력 쪽 연산을 매번 다시 하지 않아도 된다.  

---

### 3. Example SPEN Architecture

SPEN의 특성 중 feature network $$F(x) = g(A_2 g(A_1 x))$$는 2-layer MLP로 정의된다. 여기서 $g(\cdot)$는 좌표별 비선형 함수이며, 실험에서는 위치에 따라 sigmoid, ReLU, HardTanh 등을 다르게 사용할 수 있다.  
에너지 네트워크는 크게 local energy와 global energy의 합으로 구성된다.  

Local energy에 대한 각 레이블을 독립적으로 점수화하는 항은 $$E_x^{\text{local}}(\bar y) = \sum_{i=1}^{L} \bar y_i\, b_i^\top F(x)$$으로 표현할 수 있다. 이는 라벨별 독립적인 선형 score를 더한 형태이며, 각 라벨을 개별적으로 평가하는 항임을 의미한다. 이로 인해 SPEN의 성능을 평가할 때에는 global term이 얼마나 구조 정보를 추가하는지로 다뤄야 한다.  

Global energy에서 레이블 간 상호작용을 담당하는 항은 $$E_x^{\text{label}}(\bar y) = c_2^\top g(C_1 \bar y)$$이다. 여기서 $C_1 \bar y$는 출력 레이블들의 학습된 affine measurement으로 몇 개의 측정값을 만드는 원리를 갖는다. 이는 레이블을 단순히 pairwise graph로 연결하는 대신, 레이블 전체를 몇 개의 저차원 측정값으로 압축한 후 비선형 변환을 적용하는 구조다. 이 과정에서 각 hidden unit은 $\bar y$가 여러 좌표를 한꺼번에 보며 특정 패턴을 감지하는 역할을 한다. 해당 원리는 레이블 간 의존성을 사전에 그래프로 고정하지 않아도 되며, 고차 상호작용을 pairwise CRF보다 훨씬 유연하게 다루므로 더 복잡한 조합을 표현할 수 있다. 또한, 선형 스케일링적 측면에서 파라미터 수가 $L$에 대해 선형적으로 증가하기에 $L^2$보다 더 유리하며, 측정 행렬 $C_1$을 해석하면 structure learning 결과를 직접 들여다볼 수 있다는 특징이 있다.  

앞선 수식을 입력과 출력을 함께 쓰는 조건부 global energy로 수식을 변형할 수 있다.  

$$E_x^{\text{cond}}(\bar y) = d_2^\top g(D_1[\bar y; F(x)])$$

정리하면, 이 아키텍처는 입력-레이블 정합성은 local energy로, 레이블-레이블 상호작용은 global energy로 나누어 모델링한다. 이 분해가 SPEN의 해석성과 확장성을 동시에 만든다.

#### 3.1. Conditional Random Fields as SPENs

저자들은 SPEN을 이해하기 쉽게 하기 위해 pairwise CRF와의 관계를 짚는다. 완전연결 pairwise CRF를 생각하면, 데이터 의존 unary와 데이터 비의존 pairwise 항을 갖는 에너지는 대략 다음 꼴이 된다.

$$E_x^{\text{crf}}(y) = y^\top S_1 y + s^\top y$$

이 표현은 직관적이지만, 레이블 수가 커지면 즉시 문제가 드러난다.

- pairwise 상호작용을 위해 파라미터 수가 최소 $O(L^2)$로 증가한다.
- 추론 비용 역시 빠르게 커진다.
- practitioners가 label dependency를 미리 알아야 sparse graph를 설계할 수 있다.
- pairwise를 넘어서는 고차 상호작용은 훨씬 비싸다.

SPEN은 이런 제약을 우회한다. 특히 $C_1$의 affine measurement는 단순한 low-rank pairwise factorization보다 더 넓은 종류의 label interaction을 담을 수 있으며, 레이블 간 dissociativity 같은 특성도 포착할 수 있다. 물론 CRF의 표현력은 그래프 구조와 조건부 분포 클래스 사이의 관계가 잘 알려져 있지만, SPEN은 사용한 딥 네트워크의 일반적 표현력에 의존하므로 이론적 분석은 더 어렵다.

---

### 4. Learning SPENs

예측이 $\bar y$에 대한 연속 최적화라면, 학습 역시 그 구조를 고려해야 한다. 논문은 이를 위해 Structured SVM(SSVM) 기반 학습을 사용한다. 직관적으로는 정답의 에너지가 오답보다 일정 마진만큼 낮아지도록 학습하는 방식이다.

오차 함수 $\Delta(y_p, y_g)$를 Hamming loss 같은 structured loss라고 하면, 학습 목적은 다음과 같다.

$$\sum_{(x_i, y_i)}
\left[
\max_{y}
\left(
\Delta(y_i, y) - E_{x_i}(y) + E_{x_i}(y_i)
\right)
\right]_+$$  

여기서 $[\cdot]_+ = \max(0,\cdot)$ 이고, 논문은 예측이 근사 최적화이기 때문에 이 hinge 형태를 명시적으로 유지한다. SPEN에서 실제로 필요한 내부 문제는 loss-augmented inference이다.

$$y_p = \arg\min_{y}
\left(
- \Delta(y_i, y) + E_{x_i}(y)
\right)$$  

하지만 SPEN은 이 문제 역시 이산 조합 탐색으로 풀지 않고, $\bar y \in [0,1]^L$에 대한 gradient descent로 근사한다. 이때 Hamming loss는 미분 가능하지 않으므로, squared loss나 log loss 같은 미분 가능한 surrogate loss로 대체한다. 특히 log loss는 경계에서 발산하므로, entropic mirror descent가 여기서도 중요한 역할을 한다.

학습 과정의 핵심 포인트는 다음과 같다.

- SSVM 목적은 정답 에너지와 예측 에너지의 간격을 키우는 데 집중한다.
- 따라서 loss-augmented inference의 결과를 굳이 0-1로 rounding하지 않고, 연속 완화 해 $\bar y$ 자체를 사용해도 괜찮다.
- gradient는 에너지 네트워크를 통해 backpropagation으로 계산한다.
- 예측이 비볼록 최적화의 근사해라는 점은 약점이지만, 저자들은 이것이 실용적으로 충분히 잘 동작한다고 본다.

또한 저자들은 학습 안정화를 위해 다음과 같은 실전 전략을 권한다.

- 먼저 local label-wise loss만으로 feature network를 pretraining한다.
- 데이터가 적을 때는 global energy를 학습하는 동안 feature network를 고정해 과적합을 줄인다.
- 마지막 단계에서 전체 네트워크를 작은 learning rate로 함께 미세조정한다.

이 부분은 SPEN이 단순히 "출력에 대해 gradient descent를 돌린다"는 수준이 아니라, structured margin 학습과 iterative inference를 일관되게 묶어낸 모델임을 보여준다.

---

### 5. Applications of SPENs

논문이 가장 강하게 겨냥하는 응용은 multi-label classification이다. 데이터는 다음 형태를 가진다.

$$(x, y), \qquad
y = \{y_1, \dots, y_L\} \in \{0,1\}^L$$  


여기서 각 label은 독립이 아니라 서로 상관되거나 배타적일 수 있다. 문제는 이런 상호작용 구조가 주어지지 않는 경우가 많다는 점이다. SPEN은 바로 이 부분에서 장점을 가진다. (5)의 측정 행렬 $C_1$을 학습함으로써, 레이블 간 구조를 데이터에서 자동으로 발견할 수 있기 때문이다.

더 넓게 보면 SPEN은 MAP inference 문제로 쓸 수 있는 거의 모든 structured prediction 문제에 적용 가능하다. 예를 들어 시퀀스 라벨링처럼 temporal structure가 있는 경우에는 $C_1$을 반복 블록 대각 구조나 temporal convolution으로 설계해, 도메인 지식을 부드럽게 반영할 수 있다. 즉, SPEN은 완전히 무구조한 접근이 아니라, 필요하면 도메인 구조를 신경망 파라미터화 안에 녹여 넣는 유연한 틀이라고 보는 편이 맞다.

---

### 6. Related Work

#### 6.1. Multi-Label Classification

가장 단순한 다중 레이블 분류는 각 label $y_i$를 독립적으로 예측하는 binary relevance 방식이다. 구현은 간단하지만, 레이블 간 상관이나 희귀 label에 대한 보완이 불가능하다는 한계가 있다. 이를 보완하기 위해 ranking loss나 max-margin 학습을 사용하는 방법들이 제안되었고, 또 다른 흐름으로는 저차원 label embedding이나 low-rank parameterization을 이용하는 방식이 발전해 왔다.

SPEN의 관점에서 보면, 입력 표현만 저차원으로 압축하고 global energy를 쓰지 않는 선형 혹은 얕은 모델은 계산량 면에서는 매우 효율적이다. 그러나 이런 모델은 mutual exclusivity나 implicature 같은 강한 구조 제약을 표현하기 어렵다. 반면 SPEN은 non-linear feature network와 global energy를 함께 사용하여, 저차원 구조를 가지면서도 레이블 간 제약을 모델링할 수 있다.

또한 기존 structured prediction 기반 multi-label 모델은 CRF류 구조를 사용하면서, 레이블 수가 커질수록 파라미터 수와 추론 비용이 급격히 증가하거나, classifier chains처럼 강한 분해 가정을 둬야 하는 문제가 있었다. 본 논문의 global energy는 compressed sensing 기반 multi-label learning에서 영감을 받았지만, 그보다 더 일반적이다. 즉, 레이블 희소성을 강하게 가정하지 않고도 학습된 measurement + nonlinearity로 구조를 표현한다.

#### 6.2. Deep Structured Models

딥러닝과 구조화 예측의 결합은 크게 두 방향으로 발전해 왔다.

- CRF의 unary/pairwise potential을 deep feature로 파라미터화하는 방식
- approximate inference 알고리즘을 여러 단계 unroll한 뒤, 전체를 computation graph로 보고 end-to-end로 학습하는 방식

이 두 흐름은 모두 강력하다. 특히 입력 $x$에 대한 표현 학습 능력은 탁월하다. 하지만 출력 $y$에 대한 구조는 여전히 underlying graphical model에 의해 제한된다. 즉, deep network가 더해져도 출력 구조의 형태 자체는 미리 정한 모델의 범위를 벗어나기 어렵다.

SPEN은 여기서 한 걸음 더 나간다. 출력을 그래프 위 변수로 고정하지 않고, 에너지 함수 전체를 딥 네트워크로 두기 때문이다. 덕분에 모델링은 훨씬 유연해지지만, 반대로 inference는 더 이상 그래프별 최적화 알고리즘의 보장을 받지 못한다.

#### 6.3. Iterative Prediction using Neural Networks

일반적인 딥러닝에서는 backpropagation이 네트워크 파라미터를 업데이트하는 데 쓰인다. 그러나 SPEN에서는 역전파가 출력 변수 $\bar y$를 직접 최적화하는 예측 알고리즘으로 쓰인다. 저자들은 이것이 완전히 새로운 발상은 아니지만, 구조화 객체 예측에 본격적으로 적용된 예는 많지 않았다고 본다.

관련 예로는 다음과 같은 작업들이 거론된다.

- adversarial example 생성
- 문서/샘플 embedding 최적화
- 이미지 합성 및 texture synthesis
- iterative error feedback 기반 pose estimation

즉, SPEN은 "예측 = forward pass 한 번"이라는 관습 대신, 예측 자체를 differentiable optimization loop로 취급하는 broader 흐름 안에 위치한다.

---

### 7. Experiments

#### 7.1. Multi-Label Classification Benchmarks

이 절에서는 SPEN을 표준 다중 레이블 벤치마크에 적용해 BR, LR, MLP, DMF와 비교한다.

- BR: label별 독립 logistic regression
- LR: low-rank weights 기반 방법
- MLP: feature network + local energy만 있는 feed-forward baseline
- DMF: fully connected pairwise CRF에서 mean-field inference를 5단계 unroll한 deep mean field baseline
- SPEN: local + global energy를 함께 쓰는 제안 모델

논문은 example-averaged F1을 사용해 성능을 비교한다.

| Dataset | BR | LR | MLP | DMF | SPEN |
|---|---:|---:|---:|---:|---:|
| Bibtex | 37.2 | 39.0 | 38.9 | 40.0 | 42.2 |
| Bookmarks | 30.7 | 31.0 | 33.8 | 33.1 | 34.4 |
| Delicious | 26.5 | 35.3 | 37.8 | 34.2 | 37.5 |

표의 결과는 몇 가지 중요한 사실을 보여준다.

첫째, SPEN은 세 데이터셋 중 두 개에서 최고 성능을 기록하며, structured baseline인 DMF보다도 강하다. 특히 pairwise potential 수가 많은 DMF는 완전연결 CRF를 쓰기 때문에 통계적으로 과적합되기 쉬웠고, Bookmarks와 Delicious에서는 이 문제가 크게 드러난다.

둘째, 단순해 보이는 MLP baseline이 생각보다 매우 강하다. 논문은 이것을 중요한 관찰로 본다. 즉, 출력 구조를 항상 iterative inference로 풀어야만 하는 것은 아니며, 충분히 큰 feed-forward 모델도 상당한 구조를 흡수할 수 있다.

셋째, Delicious에서는 MLP가 SPEN보다 근소하게 좋다. 저자들의 해석은 명확하다. Delicious에서는 thresholding에 앞서 잘 calibration된 soft prediction이 중요한데, logistic loss로 훈련된 MLP가 margin-based SPEN보다 이 부분에서 유리했다. 그래서 SPEN은 test time에 entropy smoothing을 추가해 soft prediction을 더 부드럽게 만들어야 했다.

또한 논문은 missing label setting을 다루는 다른 multi-label 연구들과 SPEN이 경쟁 관계가 아니라 상보적이라고 본다. 즉, SPEN의 구조 학습 틀 위에 missing data 처리 기법을 결합할 여지가 있다.

#### 7.2. Comparison to Alternative SSVM Approaches

SPEN의 학습은 SSVM 기반이므로, train/test inference가 정확하지 않을 때 어떤 일이 벌어지는지도 중요하다. 이를 보기 위해 논문은 label 수가 14개인 Yeast 데이터셋에서 다음 예측 방식을 비교한다.

| Method | EXACT | LP | LBP | DMF | SPEN |
|---|---:|---:|---:|---:|---:|
| Hamming error | 20.2 ± .5 | 20.5 ± .5 | 24.3 ± .6 | 23 ± .2 | 20.0 ± .3 |

이 결과는 두 가지 관점에서 의미가 있다.

첫째, DMF가 loopy graphical model을 위한 baseline으로 완전히 부적절한 것은 아니며, LBP보다 나은 성능을 낸다. 따라서 7.1절에서 DMF를 structured deep baseline으로 둔 선택은 어느 정도 정당화된다.

둘째, 더 중요한 점은 SPEN이 EXACT나 LP와 비슷한 수준까지 도달한다는 사실이다. 이는 비볼록 최적화 기반의 근사 추론이 항상 SSVM 학습을 망치지는 않음을 시사한다. 저자들은 이를 두고, 적어도 본 실험에서는 SPEN의 inexact inference가 LBP처럼 심각한 under-generation 문제를 일으키지 않았다고 해석한다.

물론 pairwise CRF와 SPEN은 표현력이 다르므로, 이 결과만으로 학습 알고리즘의 효과를 완전히 분리해서 말할 수는 없다. 그럼에도 불구하고, "inference가 근사적이기 때문에 SSVM 학습이 반드시 취약하다"는 우려를 SPEN이 상당 부분 누그러뜨린다는 메시지는 분명하다.

#### 7.3. Structure Learning Using SPENs

이 절은 SPEN의 가장 매력적인 성질 중 하나인 구조 해석 가능성을 보여준다. 저자들은 레이블이 4개씩 묶인 블록 내부에서 서로 배타적인 synthetic task를 만든다.

데이터 생성 절차는 다음과 같다.

1. $64$차원 feature를 갖는 설계 행렬 $X$를 표준정규분포에서 샘플링한다.
2. $64 \times 16$ 가중치 행렬 $A$를 다시 샘플링한다.
3. $Z = XA$를 만든 뒤, 16개 label을 4개씩 연속 블록으로 나눈다.
4. 각 블록에서 가장 큰 값의 위치만 1로 두고 나머지는 0으로 둔다.

즉, 각 블록마다 정확히 하나의 label만 활성화되므로, 데이터에는 강한 within-block mutual exclusivity가 숨어 있다.

이때 global energy의 측정 행렬 $C_1$을 들여다보면, SPEN이 어떤 레이블들을 함께 보고 있는지 직접 관찰할 수 있다. 논문은 특히 비선형 함수 선택에 따라 해석 가능성이 달라짐을 보여준다.

- ReLU 사용 시: 여러 hidden unit이 일부 label에 큰 가중치를 나누어 가지며, 제약 위반 탐지가 hidden unit들의 선형 결합에 분산된다. 구조는 학습되지만 해석은 덜 직관적이다.
- HardTanh 사용 시: 각 measurement가 포화 특성을 가지므로, 블록 구조를 보다 개별적이고 선명하게 포착한다. 결과적으로 측정 행렬의 패턴이 훨씬 해석 가능해진다.

성능 비교는 다음과 같다.

| # train examples | Linear | 3-Layer MLP | SPEN |
|---|---:|---:|---:|
| 1.5k | 80.0 | 81.6 | 91.5 |
| 15k | 81.8 | 96.3 | 96.7 |

여기서 핵심 메시지는 분명하다.

- 적은 데이터에서는 SPEN이 압도적이다.  
  전역 구조를 반영하는 global energy가 훨씬 더 압축적(parsimonious)으로 설계되어 있기 때문이다.
- 데이터가 충분히 많아지면 MLP도 따라온다.  
  처음에는 mutual exclusivity 같은 강한 제약은 iterative inference가 필요할 것처럼 보였지만, 충분히 큰 feed-forward network도 이를 근사적으로 학습할 수 있었다.

즉, 이 절은 SPEN의 장점을 "무조건 더 표현력이 세다"가 아니라, 구조를 더 적은 파라미터로 더 빨리 학습할 수 있다는 방향으로 보여준다.

#### 7.4. Convergence Behavior of SPEN Prediction

SPEN의 test-time prediction은 gradient-based iterative optimization이므로, 실제로 얼마나 빨리 수렴하는지와 batch 처리에서 어떤 병목이 생기는지가 중요하다. 저자들은 GPU에서 큰 minibatch로 예측을 수행할 때 curse of the last reducer 문제가 발생한다고 지적한다. 즉, 대부분의 샘플이 이미 수렴했더라도, 가장 늦게 수렴하는 몇 개 때문에 전체 batch가 계속 계산을 수행하게 된다.

Bibtex 데이터셋에서의 관찰은 다음과 같다.

- 대부분의 예시는 약 20 step 부근에서 수렴한다.
- 가장 늦은 예시는 41 step 정도까지 필요하다.
- 최적화 대상은 $[0,1]^L$ 위의 비볼록 에너지이지만, 실제 예측은 종종 0 또는 1 부근으로 뾰족하게 모인다.

저자들은 속도 향상을 위해 두 가지 조기 종료 전략을 실험한다.

1. batch의 일정 비율이 수렴하면 종료
2. 수렴 tolerance를 느슨하게 설정

두 경우 모두, 정확도 저하를 거의 일으키지 않으면서 약 3배 수준의 속도 향상을 얻는다. 특히 느리게 수렴하는 샘플은 원래도 잘못 예측될 가능성이 높다는 가설이 제시된다. 또한 MLP의 출력을 초기값으로 사용하면 수렴 곡선이 대략 5 iteration 정도 왼쪽으로 이동해 더 빨라진다.

논문은 MLP와 SPEN의 실제 속도 차이도 보여준다. Bibtex test set 전체 2515개 예측에 대해 MLP는 사실상 즉시 끝나는 반면, SPEN은 iterative optimization 때문에 훨씬 느리다. 그럼에도 feature network는 한 번만 계산하면 되고, 추가 비용은 주로 출력 최적화에 집중되므로, 표현력과 속도의 절충이 비교적 명확한 모델이라고 볼 수 있다.

마지막으로 저자들은 search error도 측정한다. 즉, 최적화가 반환한 해의 에너지가 ground truth에서의 에너지보다 더 큰 경우를 세어 보면, Bibtex에서 약 8% 정도 이런 실패가 발생한다. 이는 SPEN의 가장 본질적인 한계, 즉 강한 모델링 유연성을 지역 최적화 기반 추론과 맞바꾸었다는 사실을 상기시킨다.

---

### 8. Conclusion and Future Work

이 논문은 structured prediction에서 딥러닝을 입력 표현 학습에만 쓰지 않고, 출력 구조 자체를 에너지 함수로 표현하는 방향을 제시했다는 점에서 의미가 크다. SPEN은 예측을 gradient-based optimization으로 수행하기 때문에, 기존 그래프 구조에 맞춰 inference algorithm을 새로 설계하지 않고도 고차 상호작용을 포함하는 유연한 에너지 함수를 사용할 수 있다.

특히 논문이 보여준 핵심 메시지는 다음과 같이 정리할 수 있다.

- 적은 데이터에서는 잘 설계된 energy-based 모델이 더 압축적이어서 유리할 수 있다.
- multi-label classification처럼 label topology가 주어지지 않는 문제에서 SPEN은 구조 학습 도구로 자연스럽다.
- iterative prediction은 느리지만, 그 대가로 feed-forward 모델보다 더 직접적으로 구조를 주입하고 해석할 수 있다.

저자들이 제안한 후속 연구 방향도 설득력 있다. 하나는 $y$에 대해 convex한 SPEN을 설계해 추론 안정성을 높이는 것이고, 다른 하나는 gradient-based prediction 과정 자체를 학습에 포함하여 prediction procedure까지 end-to-end로 최적화하는 것이다. 즉, 본 논문은 완성형 해답이라기보다, 딥 structured prediction을 훨씬 넓은 함수 공간으로 확장하는 출발점에 가깝다.

---

### A. Appendix

부록은 본문에서 다룬 아이디어를 더 실무적이고 알고리즘적인 수준으로 보강한다. 특히 수렴 특성, DMF baseline의 구체식, 실전 학습 팁, 하이퍼파라미터가 정리되어 있어 재현 관점에서 중요하다.

#### A.1. Analysis of Convergence Behavior

부록의 Figure 2, Figure 3, Figure 4는 SPEN 예측의 실제 수렴 양상을 시각적으로 보여준다.

- Figure 2: batch 안의 대부분 예시는 빠르게 수렴하고, 소수의 느린 예시가 전체 계산 시간을 끌어올린다.
- Figure 3: 전체 예시 중 일정 비율만 수렴해도 종료하도록 하면, 정확도 손실이 거의 없으면서 큰 속도 향상을 얻는다.
- Figure 4: convergence tolerance를 느슨하게 하면 속도는 더 빨라지고, 정확도는 조금 감소한다.

즉, SPEN prediction의 실전 병목은 평균 샘플이 아니라 최악의 수렴 사례이며, 논문은 이 점을 이용해 실용적인 early stopping 전략을 제안한다.

#### A.2. SPEN Architecture for Multi-Label Classification

부록의 계산 그래프는 본문 3절의 구조를 한 장으로 요약한다.

1. 입력 $x$는 feature network를 지나 $F(x)$가 된다.
2. 출력 후보 $\bar y$는 local energy와 global energy에 동시에 입력된다.
3. 두 에너지 항을 더해 최종 scalar energy를 만든다.
4. 예측 시에는 이 scalar를 줄이는 방향으로 $\bar y$를 업데이트한다.

즉, SPEN은 "입력에서 출력을 한 번에 뽑는 네트워크"라기보다, 출력 공간을 탐색하기 위한 미분 가능한 에너지 지형을 학습하는 네트워크라고 이해하는 편이 정확하다.

#### A.3. Deep Mean Field Predictor

논문은 비교 실험을 위해 fully-connected pairwise CRF의 mean-field inference를 unroll한 Deep Mean Field(DMF) baseline을 직접 구성한다. 시작점은 다음 확률모형이다.

$$P(y \mid x) \propto
\exp\left(
\sum_{i,j} B_{ij}^{(x)}(y_i, y_j)
+
\sum_i U_i^{(x)}(y_i)
\right)$$

이를 이진 벡터 $y \in \{0,1\}^L$의 형태로 정리하면, pairwise 및 unary 항을 모아 다음처럼 쓸 수 있다.

$$P(y \mid x) \propto
\exp\left(
y^\top A_1 y + (1-y)^\top A_2 y + (1-y)^\top A_3(1-y)
+ C_1^\top y + C_2^\top (1-y)
\right)$$

그리고 다시 정리하면 결국

$$P(y \mid x) \propto \exp\left(y^\top A y + C^\top y\right)$$

의 형태가 된다.

이제 mean-field에서 $\bar y_t \in [0,1]^L$를 시점 $t$의 marginal estimate라고 두면, 각 좌표 업데이트는 다음 sigmoid 형태로 나타난다.

$$\bar y_i^{t+1}=\frac{\exp(e_i^1)}{\exp(e_i^1)+\exp(e_i^0)}=\mathrm{Sigmoid}(e_i^1 - e_i^0)$$  

여기서 계산을 전개하면

$$e_i^1 - e_i^0 = s_i + C_i$$  

꼴이 되어, 전체 업데이트를 벡터화할 수 있다. 논문이 제시한 알고리즘은 아래와 같다.

```text
Algorithm 1 Vectorized Mean-Field Inference for Fully-Connected Pairwise CRF for Multi-Label Classification

Input: x, m
A, C <- GetPotentials(x)
Initialize y_bar uniformly as [0.5]^L
D <- diag(A)

for t = 1 to m do
    E <- A y_bar - D + C
    y_bar <- Sigmoid(E)
end for
```

이 baseline은 중요한 비교 기준이다. 왜냐하면 "CRF 계열 structured model을 iterative neural computation으로 바꾼 방식"과 "아예 에너지 자체를 neural network로 둔 방식"을 정면 비교할 수 있게 해 주기 때문이다.

#### A.4. Details for Improving Efficiency and Accuracy of SPENs

부록은 SPEN을 실제로 잘 동작시키기 위한 실용적 요령도 정리한다.

- Momentum 사용: prediction-time optimization에도 momentum을 넣어 비볼록 목적함수에서 수렴을 개선한다.
- GPU minibatch inference: 여러 예시를 병렬로 풀어 계산 효율을 높인다.
- Entropy augmentation: soft prediction이 중요할 때는 test time에 $\bar y$의 entropy를 더해 분포를 부드럽게 만든다.
- 출력 쪽 gradient만 계산: inference에서는 파라미터 gradient가 필요 없으므로, $\bar y$에 대한 gradient만 계산해 속도를 줄인다.
- 단계적 학습:  
  1) local energy를 먼저 학습하고  
  2) local을 고정한 채 global energy를 학습한 뒤  
  3) 마지막에 전체를 작은 learning rate로 joint fine-tuning 한다.

즉, SPEN의 성능은 모델 아이디어 자체뿐 아니라 최적화 세부 구현에도 크게 좌우된다.

#### A.5. Hyperparameters

논문이 보고한 주요 설정은 다음과 같다.

- Prediction
  - gradient descent with momentum = 0.95
  - learning rate = 0.1
  - learning rate decay 없음
  - 종료 조건: 목적함수의 상대 변화량이 작아지거나, iterate의 $l_\infty$ 변화량이 충분히 작아질 때

- Training
  - SGD with momentum = 0.9
  - learning rate와 decay는 validation으로 조정
  - pretraining과 SSVM training 모두에서 $l_2$ regularization 사용

- Network size
  - feature network hidden size는 exhaustive search로 맞춘 것이 아니라, 데이터 크기와 직관을 바탕으로 정했다.

부록의 데이터셋 요약표는 아래와 같다.

| Dataset | #labels | #features | # train | % true labels |
|---|---:|---:|---:|---:|
| Bibtex | 159 | 1836 | 4880 | 2.40 |
| Delicious | 983 | 500 | 12920 | 19.02 |
| Bookmarks | 208 | 2150 | 60000 | 2.03 |
| Yeast | 14 | 103 | 2417 | 30.3 |

이 수치는 본문 7절의 결과를 해석할 때 중요하다. 예를 들어 Delicious는 label 수가 매우 크고 positive density도 비교적 높아, 잘 calibration된 soft prediction이 특히 중요해지는 반면, Bibtex와 Bookmarks는 더 희소한 label 공간에서 구조 학습의 효과가 드러나기 쉽다.

---

**Review by 변정우, Aerospace Engineering Undergraduate Researcher**  
**[Update - Time Log]**  
* 2026.03.18: [Draft] 1-4 파트 리딩 완료
* 2026.03.21: [ver_1] 1-2 파트 글 작성
* 2026..: [ver_2]
* 2026..: [Final_ver]
