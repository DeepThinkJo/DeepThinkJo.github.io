---
title: "ML Practice - Iris Species Classification with Softmax Regression"
date: 2026-11-10
categories: [Machine Learning, Practice]
tags: [AI, ML, supervised-learning, softmax-regression, practice, computational-graph]
math: true
---

## Iris Species Classification with Softmax Regression

Softmax regression의 학습 과정을 직접 구현하며 복습한 내용을 정리한 포스트이다.
Iris dataset을 이용해 softmax regression model을 구현하고, batch gradient descent를 통해 최적의 parameter를 구한다.
Parameter에 대한 loss function의 gradient를 구하는 과정이 자세히 서술되어있다.

## 1. Iris Dataset

이 실습에 쓰인 dataset은 Kaggle에서 다운받은 'Iris.csv' 파일이다.
이 파일은 'SepalLengthCm' (꽃받침의 길이), 'SepalWidthCm' (꽃받침의 두께), 'PetalLengthCm' (꽃잎의 길이), 'PetalWidthCm' (꽃잎의 두께), 그리고 'Species' (꽃의 종류)의 5가지 column을 갖는다. Example은 총 150개가 있다.

이때 'Species' column은 다음의 세 가지 값 중 하나를 가진다.

1. "Iris-setosa"
2. "Iris-versicolor"
3. "Iris-virginica"

즉 어떤 붓꽃의 꽃받침과 꽃잎 각각에 대한 길이와 두께가 주어졌을 때 그 붓꽃의 종류를 위 세 가지 값 중 하나로 예측하는 **multi-class classification** 문제다.

이 dataset을 활용하여 softmax regression model을 training하기 위한 사전 작업을 해야 한다.

먼저 feature matrix의 왼쪽에 모든 element가 1인 column을 추가하여 design matrix를 구성한다.

즉, 주어진 dataset의 feature matrix를 아래와 같은 $150 \times 4$ matrix라고 생각하면,

$$
X_\text{feature} =
\begin{bmatrix}
x_1^{(1)}    & x_2^{(1)}   & x_3^{(1)}    & x_4^{(1)}  \\
x_1^{(2)}    & x_2^{(2)}   & x_3^{(2)}    & x_4^{(2)}  \\
\vdots       & \vdots      & \vdots       & \vdots     \\
x_1^{(150)}  & x_2^{(150)} & x_3^{(150)}  & x_4^{(150)}
\end{bmatrix}
\in \mathbb{R}^{150 \times 4}
$$

design matrix를 다음과 같이 만드는 것이다.

$$
X =
\begin{bmatrix}
1      & x_1^{(1)}    & x_2^{(1)}   & x_3^{(1)}    & x_4^{(1)}  \\
1      & x_1^{(2)}    & x_2^{(2)}   & x_3^{(2)}    & x_4^{(2)}  \\
\vdots & \vdots       & \vdots      & \vdots       & \vdots     \\
1      & x_1^{(150)}  & x_2^{(150)} & x_3^{(150)}  & x_4^{(150)}
\end{bmatrix}
\in \mathbb{R}^{150 \times 5}
$$

또한 원본 dataset의 각 example $i$의 label, $y^{(i)}$를 다음의 규칙에 따라 **one-hot encoding**하자.

$$
\mathbf{y}^{(i)} =
\begin{cases}
    \begin{bmatrix}
        1 & 0 & 0
    \end{bmatrix}, & y_\text{original}^{(i)}=\text{Iris-setosa} \\
    \begin{bmatrix}
        0 & 1 & 0
    \end{bmatrix}, & y_\text{original}^{(i)}=\text{Iris-versicolor} \\
    \begin{bmatrix}
        0 & 0 & 1
    \end{bmatrix}, & y_\text{original}^{(i)}=\text{Iris-virginica}
\end{cases}
$$

이와 같이 고유한 category의 개수만큼 새로운 column을 만들어, 어떤 example이 하나의 category에 속하면 그 column에는 1을 할당하고, 나머지 category에 대응되는 column들에는 모두 0을 할당하는 data preprocessing 과정을 *one-hot encoding*이라 한다. One-hot encoding을 하는 이유는, 서로 다른 category 간 의도치 않은 수학적 순서나 순위를 매기지 않기 위해서이다.


## 2. Notations

본격적으로 문제를 해결하기 전, 표기를 확립하자.

| Notation                                           | Meanings                                    |
| -------------------------------------------------- | ------------------------------------------- |
| $m$                                                | Example의 개수 (이 문제의 경우 $m=150$)     |
| $n$                                                | Feature의 개수 (이 문제의 경우 $n=4$)       |
| $K$                                                | Class의 개수 (이 문제의 경우 $K=3$)         |
| $\mathbf{x}^{(i)} \in \mathbb{R}^{1 \times (n+1)}$ | Input Row Vector (가장 첫 index에 1 첨가됨) |
| $\mathbf{y}^{(i)} \in \set{0, 1}^{1 \times K}$     | Label of example $i$ (One-hot encoded)      |
| $\hat{\mathbf{y}}^{(i)} \in (0, 1)^{1 \times K}$   | Example $i$에 대한 model의 prediction       |
| $X \in \mathbb{R}^{m \times (n+1)}$                | Design matrix                               |
| $Y \in \set{0, 1}^{m \times K}$                    | Target matrix                               |
| $\hat{Y} \in (0, 1)^{m \times K}$                  | Prediction Matrix                           |
| $\Theta \in \mathbb{R}^{(n+1) \times K}$           | Parameter                                   |
| $J(\Theta)$                                        | Loss function                               |

## 3. Softmax Regression Model

Softmax regression model은 주어진 input이 각각의 class $j=1,\ldots,K$에 속할 확률을 예측한다.

따라서 softmax regression model이 output으로 내놓는 값은 $1 \times K$ row vector이고, 각 element $j$는 input이 class $j$에 속할 확률을 나타낸다.

위 목적에 맞게 output을 내놓는 function이 필요한데, 그것이 바로 softmax function이다. Softmax function은 다음과 같이 정의된다.

$$
\operatorname{softmax}(\mathbf{z}) =
\begin{bmatrix}
\frac{\operatorname{exp}(z_1)}{\sum_{j=1}^K \operatorname{exp}(z_j)} &
\frac{\operatorname{exp}(z_2)}{\sum_{j=1}^K \operatorname{exp}(z_j)} &
\cdots &
\frac{\operatorname{exp}(z_K)}{\sum_{j=1}^K \operatorname{exp}(z_j)}
\end{bmatrix}
$$

즉, 어떤 vector에 element-wise하게 exponential function을 취한 후, 그들의 합으로 나누는 기능을 한다.

Exponential function을 취하는 이유는 여러가지가 있다.
그 중 하나는, exponential function의 output은 항상 nonnegative인 것이다. Exponential function을 취함으로써 softmax function의 output의 각 element가 음수가 될 수 없게끔 하는 것이다.

이후 $\sum_{j=1}^K \operatorname{exp}(z_j)$로 모든 element를 나눔으로써 각 element는 $0$과 $1$ 사이의 값을 갖고, 모든 element의 합이 1이 되게끔 할 수 있다.
각 확률은 $0$과 $1$ 사이의 값이어야 하고, input은 무조건 $K$개의 class 중 하나로 분류되므로 확률적으로 올바른 결과이다.

Softmax regression model은 다음과 같이 prediction 값을 반환한다.

$$
\hat{\mathbf{y}}(\mathbf{x}; \Theta) =
\operatorname{softmax}(
    \mathbf{x}\Theta
)
$$

## 4. Batch Gradient Descent

Softmax regression model의 parameter update를 위한 gradient 유도 과정은 꽤 복잡하다.
따라서 하나의 example에 대한 loss function $J^{(i)}$의 gradient를 먼저 구한 후, batch loss function에 대한 gradient를 구하자.

### 4-1. Sample Gradient

표기의 편리성을 위해 superscript $(i)$는 생략하겠다.

Sample $i$의 input을 $\mathbf{x} \in \mathbb{R}^{1\times(n+1)}$, target vector를 $\mathbf{y} \in \mathbb{R}^{1\times K}$, 그리고 prediction을 $\hat{\mathbf{y}}$라 하면, sample loss function은 다음과 같이 정의된다.

$$
J(\Theta) = -\sum_{j=1}^K y_j \operatorname{log} \hat{y}_j
$$

$J$의 $\Theta$에 대한 gradient를 바로 구하는 것은 힘드므로 chain rule을 활용하자.

먼저 $\hat{\mathbf{y}}$에 대한 downstream gradient를 구하자.

$$
\frac{\partial J}{\partial \hat{y}_j} =
-\frac{y_j}{\hat{y}_j}
$$

따라서,

$$
\nabla_{\hat{\mathbf{y}}}J =
\begin{bmatrix}
    -\frac{y_1}{\hat{y}_1} &
    -\frac{y_2}{\hat{y}_2} &
    \cdots &
    -\frac{y_K}{\hat{y}_K}
\end{bmatrix}
\in \mathbb{R}^{1 \times K}
$$

이제 $y_i$의 $z_j$에 대한 partial derivative를 구해보자.

$$
\begin{align}
    \frac{\partial \hat{y}_i}{\partial z_j} &=

    \frac{\partial}{\partial z_j}
    \frac{e^{z_i}}{\sum_{k=1}^K e^{z_k}} \\ &=

    \frac{(\frac{\partial}{\partial z_j}e^{z_i})\sum_{k=1}^K e^{z_k} - e^{z_i}(\frac{\partial}
    {\partial z_j}\sum_{k=1}^K e^{z_k}) }
    {(\sum_{k=1}^K e^{z_k})^2} \\ &=
    
    \begin{cases}
        \frac
        {e^{z_i}(\sum_{k=1}^K e^{z_k}) - (e^{z_i})^2}
        {(\sum_{k=1}^K e^{z_k})^2}, & i=j \\
        \frac
        {0\cdot(\sum_{k=1}^K e^{z_k})-e^{z_i}e^{z_j}}
        {(\sum_{k=1}^K e^{z_k})^2}, & i\neq j
    \end{cases} \\ &=

    \begin{cases}
        \hat{y}_i(1-\hat{y}_i), & i=j \\
        -\hat{y}_i\hat{y}_j, & i\neq j
    \end{cases}

\end{align}
$$

따라서,

$$
\nabla_{\mathbf{z}} \hat{y}_i =
\begin{bmatrix}
    -\hat{y}_i\hat{y}_1 &
    \cdots &
    -\hat{y}_i\hat{y}_{i-1} &
    \hat{y}_i(1-\hat{y}_i) &
    -\hat{y}_i\hat{y}_{i+1} &
    \cdots &
    -\hat{y}_i\hat{y}_K &
\end{bmatrix}
$$

따라서,

$$
\nabla_{\mathbf{z}} \hat{\mathbf{y}} =
\begin{bmatrix}
    \hat{y}_1(1-\hat{y}_1) & -\hat{y}_1\hat{y}_2 & \cdots & -\hat{y}_1\hat{y}_K \\
    -\hat{y}_2\hat{y}_1 & \hat{y}_2(1-\hat{y}_2) & \cdots & -\hat{y}_2\hat{y}_K \\
    \vdots & \vdots & & \vdots \\
    -\hat{y}_K\hat{y}_1 & -\hat{y}_K\hat{y}_2 & \cdots & \hat{y}_K(1-\hat{y}_K)
\end{bmatrix}
\in \mathbb{R}^{K \times K}
$$

위에서 구한 것은 $\mathbf{z}$에 대한 local gradient다. 이제 $\mathbf{z}$에 대한 downstream gradient를 구해보자.

$$
\begin{align}
    \nabla_{\mathbf{z}} J &=
    \nabla_{\hat{\mathbf{y}}}J \nabla_{\mathbf{z}}\hat{\mathbf{y}} \\ &=
    \begin{bmatrix}
        -\frac{y_1}{\hat{y}_1} &
        -\frac{y_2}{\hat{y}_2} &
        \cdots &
        -\frac{y_K}{\hat{y}_K}
    \end{bmatrix}
    \begin{bmatrix}
        \hat{y}_1(1-\hat{y}_1) & -\hat{y}_1\hat{y}_2 & \cdots & -\hat{y}_1\hat{y}_K \\
        -\hat{y}_2\hat{y}_1 & \hat{y}_2(1-\hat{y}_2) & \cdots & -\hat{y}_2\hat{y}_K \\
        \vdots & \vdots & & \vdots \\
        -\hat{y}_K\hat{y}_1 & -\hat{y}_K\hat{y}_2 & \cdots & \hat{y}_K(1-\hat{y}_K)
    \end{bmatrix} \\ &=

    \begin{bmatrix}
        -y_1(1-\hat{y}_1) + y_2\hat{y}_1 + \cdots + y_K\hat{y}_1 \\
        y_1\hat{y}_2 - y_2(1-\hat{y}_2) + \cdots + y_K\hat{y}_2 \\
        \vdots \\
        y_1\hat{y}_K + y_2\hat{y}_K + \cdots - y_K(1-\hat{y}_K)
    \end{bmatrix}^T \\ &=

    \begin{bmatrix}
        \hat{y}_1(y_1 + \cdots + y_K) - y_1 \\
        \hat{y}_2(y_1 + \cdots + y_K) - y_2 \\
        \vdots \\
        \hat{y}_K(y_1 + \cdots + y_K) - y_K
    \end{bmatrix}^T \\ &=

    \begin{bmatrix}
        \hat{y}_1 - y_1 & \hat{y}_2 - y_2 & \cdots \hat{y}_K - y_K
    \end{bmatrix} \\ &=

    \hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^{1 \times K}

\end{align}
$$

위에서 $y_1+\cdots+y_K=1$이 쓰일 수 있었던 이유는, $\mathbf{y}$가 one-hot vector이기 때문이다.

마지막으로 $\Theta$에 대한 loss function의 downstream gradient를 구하자.

$$
\frac{\partial J}{\partial \Theta} = \mathbf{x}^T \frac{\partial J}{\partial \mathbf{z}}
$$

이므로,

$$
\begin{align}
    \nabla_{\Theta} J &= \nabla_{\Theta} \mathbf{z} \nabla_{\mathbf{z}} J \\ &=
    \mathbf{x}^T(\hat{\mathbf{y}}-\mathbf{y})
\end{align}
$$

다시 sample $i$를 표기할 superscript $(i)$를 표기하면, 다음의 결론을 얻는다.

$$
\nabla_{\Theta} J^{(i)}(\Theta) = (\mathbf{x}^{(i)})^T(\hat{\mathbf{y}}^{(i)}-\mathbf{y}^{(i)})
$$

### 4-2. Batch Gradient

Example $i$에 대한 sample loss function을 $J^{(i)}$라 하면, batch loss function은 다음과 같이 정의할 수 있다.

$$
J(\Theta) = \frac{1}{m} \sum_{i=1}^m J^{(i)}(\Theta)
$$

따라서 3-1에서 구한 sample gradient를 평균하면 batch gradient를 얻을 수 있다.

$$
\begin{align}
    \nabla_\Theta J(\Theta) &=
    \nabla_\Theta (\frac{1}{m} \sum_{i=1}^m J^{(i)}(\Theta)) \\ &=
    \frac{1}{m} \sum_{i=1}^m (\mathbf{x}^{(i)})^T(\hat{\mathbf{y}}^{(i)}-\mathbf{y}^{(i)}) \\ &=
    \frac{1}{m} X^T(\hat{Y}-Y)
\end{align}
$$

이렇게 얻은 gradient를 batch gradient descent에 적용하여, zero matrix로 초기화한 parameter $\Theta$를 learning rate $\alpha=0.01$로 10,000번 학습시킨 결과, loss를 약 1.0986에서 약 0.1240까지 낮출 수 있었다.

