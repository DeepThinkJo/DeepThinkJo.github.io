---
title: "ML in Details - Supervised Learning"
date: 2026-10-15
categories: [ML, Supervised Learning]
tags: [AI, ML, supervised-learning]
math: true
---

## 지도학습(Supervised Learning)

Supervised learning은 machine learning에서 가장 널리 사용되는 학습 방식 중 하나이다.

Supervised learnin은 input과 target value의 쌍으로 구성된 dataset을 이용하여 input과 output 사이의 관계를 모델링하는 방식이다. 학습 과정에서는 주어진 data에 적합하도록 model의 parameter를 조정한다.

비유하자면, 학생(model)에게 수학(input과 output 사이의 pattern)을 가르치기 위해,
문제집(input)과 답지(target value)를 주고 문제와 정답 사이의 관계를 스스로 학습하도록 하는 것이다.

어떤 뜻인지 아래에서 자세히 알아보자.

## 1. Supervised Learning의 목적

Supervised learning의 목적은 주어진 dataset에서 input과 output 사이의 관계를 학습하고, 새로운 input에 대해서도 정확한 output을 예측할 수 있는 model을 만드는 것이다.

위 비유에 대응시키자면, 학생으로 하여금 문제집과 답지를 통해 수학 개념을 익히고, 새로운 문제를 풀 수 있는 능력을 갖추게끔 하는 것이다.

이때 중요한 것은 "새로운 문제를 풀 수 있는 능력"이다.

학생이 모든 문제에 대응되는 답을 외우는 것과 주어진 문제들을 풀기 위해 필요한 개념을 익히는 것은 다르다.

무슨 뜻인지 자세히 설명하겠다.

Training dataset은

$$
\mathcal{D}=\{(x^{(1)},y^{(1)}),\ldots,(x^{(m)},y^{(m)})\}
$$

와 같이 input과 output의 쌍 $m$개로 구성된다. 여기서 $x^{(i)}$는 $i$번째 example의 input이고, $y^{(i)}$는 그에 대응하는 target value이다.

우리가 원하는 최적의 hypothesis는 dataset의 각 $x^{(i)}$ 값을 넣었을 때 그 input에 대응되는 output $y^{(i)}$와 가까워야 한다.

"그렇다면 주어진 모든 data point를 정확하게 통과하는 함수를 찾으면 되지 않을까?"라고 생각할 수 있다. 하지만 training data에 대한 완벽한 적합이 새로운 data에 대한 좋은 prediction을 보장하지는 않는다. Training data의 noise나 우연한 pattern까지 학습하면 training data에서는 정확하더라도 새로운 data에서는 성능이 떨어질 수 있다. 이를 과적합(overfitting)이라고 한다.

따라서 중요한 것은 주어진 data의 정답을 그대로 재현하는 것이 아니라, 데이터에 내재한 일반적인 패턴을 학습하는 것이다.

## 2. Supervised Learning의 종류

Supervised learning은 output의 종류에 따라

회귀(regression)와 분류(classification)로 나뉜다.

Regression의 경우, hypothesis가 예측하는 output은 continuous value이다.

반대로 classification은 output이 discrete value인 경우다.

예를 들어, 사람의 키를 통해 몸무게를 예측하는 model은 regression model에 해당하며, 사람의 혈당 수치를 통해 당뇨병 여부를 예측하는 model은 classification model에 해당한다.

## 3. Supervised Learning의 Workflow

이러한 목적을 달성하기 위해 supervised learning은 일반적으로 다음과 같은 단계로 진행된다.

1. Data Collection
2. Model Selection
3. Loss Function
4. Optimization
5. Evaluation and Generalization

### 3-1. Data Collection

Supervised learning에서 가장 중요한 재료다.

학생에게 준 수학 문제집과 답지에 오류가 있거나, 문제의 수가 적거나, 특정 유형의 문제에만 편양되어 있다면 학생은 일반적인 문제 해결 능력을 기르기 어렵다. 마찬가지로 ML에서도 data의 양뿐 아니라 정확성, 다양성, 대표성이 중요하다.

따라서 양과 질을 갖춘 dataset은 성공적인 학습에 필수적이다.

### 3-2. Model Selection

Model selection은 해결하려는 문제와 데이터의 특성에 맞춰 model의 구조와 함수 형태를 결정하는 과정이다.

예를 들어 사람의 키를 이용해 몸무게를 예측한다고 가정하자. 데이터에서 키와 몸무게 사이의 관계가 대략 선형적으로 나타난다면, $y=ax+b$와 같은 linear model이 $y=\frac{a}{x}$보다 적합한 선택일 수 있다.

이 단계에서 결정하는 것은 model의 함수 형태이지, parameter인 $a$와 $b$의 구체적인 값이 아니다. 이 값들은 이후 학습 과정에서 데이터에 맞게 조정된다.

이처럼 예측하고자 하는 문제나 data의 특성을 파악하여 좋은 model을 선택하는 것이 중요하다.

Model의 예로는 linear regression, logistic regression 등이 있다.

### 3-3. Loss Function

Loss function은 model의 prediction과 target value 사이의 차이를 정량화하는 함수다. 문제의 유형과 학습 목적에 따라 적절한 loss function을 선택해야 한다.

Loss function을 정의함으로써 우리는 "Dataset의 input과 output의 관계를 가장 잘 표현하는 model 찾기"라는 목적을
"Loss function을 최소화하기"라는 더 수학적으로 정량적이고 명확한 목적으로 바꿀 수 있다.

다만 training loss를 최소화하는 것과 새로운 데이터에서 좋은 성능을 보이는 것은 서로 다른 문제다.

Loss function의 예로는 Mean Squared Error, Cross-Entropy 등이 있다.

### 3-4. Optimization

Optimization은 loss function을 최소화하는 parameter의 값을 찾는 과정이다.

이때 loss function을 최소화하는 문제는 model의 구조와 loos function의 형태에 따라 복잡해질 수 있다.

일부 단순한 model에서는 analytic solution을 구할 수 있지만, 복잡한 model에서는 optimal parameter를 직접 계산하기 어려운 경우가 많다.

따라서 Gradient Descent, Momentum, Adam과 같은 반복적 optimization algorithm을 이용해 loss function을 최소화하는 parameter를 찾는다.

### 3-5. Evaluation and Generalization

학생이 얼마나 잘 공부했는지 평가하기 위해 시험을 보듯, 학습시킨 model이 잘 작동하는지 평가할 필요가 있다.

이를 위해 dataset을 training set, validation set, test set으로 나누어 사용할 수 있다. Training set은 model의 parameter를 학습하는 데 사용하고, validation set은 model이나 hyperparameter를 선택하는 데 사용한다. 마지막으로 test set은 model 선택과 조정이 끝난 뒤 최종적인 성능을 평가하는 데 사용한다.

이처럼 학습에 사용하지 않은 데이터에 대해서도 좋은 성능을 보이는 능력을 generalization이라고 한다.

## 4. Prerequisite Mathematics for Machine Learning

수학은 machine learning의 핵심 원리를 이해하고 model을 설계하는 데 중요한 기반이다.

Real-world problem을 그 자체로 해결하는 것은 매우 힘든 일이므로,
engineer는 mathematical modeling을 통해 그 문제를 과학의 영역으로 옮긴다.

CS의 발달로 그 방대하고 복잡한 문제를 수학의 영역에서 다룰 수 있게 된 것이다.

Machine learning을 공부하기 위해 필요한 수학 선수과목들은 다음과 같다.

1. Linear algebra
2. Calculus
3. Probability and Statistics

각각 왜 필요한지 간단하게 설명하겠다.
Real-world problem은 주로 여러 요인을 포함한다. Linear algebra에 등장하는 vector와 matrix 등의 개념은 데이터를 구조적으로 표현하고, model의 계산을 간결하게 나타내는 데 유용하다. 또한 matrix operation은 현대 컴퓨터의 효율적인 병렬 계산과도 잘 맞는다.
Optimization은 target function을 최소화하거나 최대화하는 문제다. 많은 ML model에서는 미분을 이용해 함수가 변하는 방향과 정도를 파악하고, 이를 바탕으로 parameter를 조정한다. 따라서 calculus는 특히 gradient-based optimization을 이해하는 데 중요하다.
Probability and Statistics는 데이터의 uncertainty를 이해하고, input과 target value 사이의 관계를 통계적으로 모델링하는 데 중요하다. 또한 통계적 관점은 model의 일반화 성능을 분석하고 평가하는 데에도 활용된다.

## 5. Mathematical Details

앞서 살펴본 supervised learning의 workflow를 수학으로 정식화해보자.

1. Data Collection
주어진 dataset은 $m$개의 input과 output의 쌍으로 이루어져 있다.

$$
\mathcal{D}=\{(x^{(1)},y^{(1)}),\ldots,(x^{(m)},y^{(m)})\}
$$

주로 dataset은 하나의 example을 하나의 row로 갖는 table 형식을 취한다.
이때 $\mathbf{x}^{(i)} \in \mathbb{R}^n, \, \mathbf{y}^{(i)} \in \mathbb{R}^{n'}, \, i=1,\ldots,m$인 경우 $\mathcal{D}$는 $m \times (n+n')$ matrix다.

3. Model Selection
$\mathcal{D}$의 input과 target value 사이의 관계를 모델링하는 함수를 $h_\theta(x)$라 하자. 여기서 $x$는 input을, $\theta$는 parameter를 나타낸다.

4. Loss Function Design
Dataset $\mathcal{D}$에 대해 정의한 loss function을 $J(\theta)$라고 하자. 이때 $h_\theta(x)$에서 $x$는 입력 변수지만, $J(\theta)$를 계산할 때는 각 example의 input과 target value가 고정되어 있다. 따라서 loss function을 최소화하는 과정에서는 parameter $\theta$를 조정한다.

5. Optimization
목적은 최적의 hypothesis function을 찾는 것인데, model selection에서 function의 틀은 정해졌으므로, 그 목적은 최적의 parameter $\theta^{\ast}$를 찾는 것과 동치이다.

$$
\theta^*=\underset{\theta}{\arg\min} J(\theta)
$$ 

위의 $\theta^*$를 찾기 위해 선택한 optimization algorithm을 사용한다.
실제로는 optimal parameter를 해석적으로 구하기 어려운 경우가 많으므로, 선택한 optimization algorithm을 이용해 그 값을 근사적으로 찾는다.

5. Evaluation and Generalization
최종적으로 학습과 model selection에 사용하지 않은 test dataset $\mathcal{D}_{\text{test}}$을 이용해 모델의 일반적인 성능을 평가할 수 있다. 이때 평가 지표는 문제의 유형에 따라 accuracy, precision, recall 등으로 달라질 수 있다.
