---
title: "ML in Details - Linear Regression Solving Salary Problem"
date: 2026-10-25
categories: [ML, Supervised Learning]
tags: [AI, ML, supervised-learning, linear-regression, practice]
math: true
---

## Solving Salary Problem Using Linear Regression

Linear regression의 학습 과정을 직접 구현하며 복습한 내용을 정리한 포스트이다.
Salary dataset을 이용해 linear regression model을 구현하고, Gradient Descent와 Least Squares Method를 통해 최적의 parameter를 구한다.
또한 두 방법의 결과를 비교하여 numerical optimization algorithm의 학습 결과를 검증한다.

## 1. Salary Dataset

이 실습에 쓰인 dataset은 'Kaggle'이란 플랫폼에서 다운받았다.
Kaggle은 data science 분야의 커뮤니티 및 경진대회 플랫폼이다.

위 사이트에서 다운받은 데이터셋은 csv 확장자 파일이다.
CSV는 Comma-Separated Values의 약자로, 데이터를 쉼표로 구분하여 저장하는 텍스트 파일 형식이다.

이 'Salary.csv'는 'YearsExperience'와 'Salary' 두 column을 갖고 있다.
또한 연차와 월급의 쌍은 총 35개가 있다.
즉 근무 경력(YearsExperience)를 활용하여 급여(Salary)를 예측하는 regression 문제다.

아래는 이 dataset의 각 data sample을 2D-plane에 plot한 이미지이다.

![Salary vs Experience Plot](/assets/img/2026-10-25/dataset_plot.png)

Linear regression 모델은 input feature의 coefficient 뿐만 아니라 bias term도 parameter로 갖는다. 이를 matrix operation으로 통일해 표현하기 위해, input feature의 왼쪽에 모든 element가 1인 column을 추가하여 design matrix를 구성한다.

즉, 주어진 dataset을 아래와 같은 $35 \times 2$ matrix라고 생각하면,

$$
D =
\begin{bmatrix}
x^{(1)} & y^{(1)} \\
x^{(2)} & y^{(2)} \\
\vdots & \vdots \\
x^{(35)} & y^{(35)}
\end{bmatrix}
\in \mathbb{R}^{35 \times 2}
$$

design matrix를 다음과 같이 만드는 것이다.

$$
X =
\begin{bmatrix}
1.0 & x^{(1)} \\
1.0 & x^{(2)} \\
\vdots & \vdots \\
1.0 & x^{(35)}
\end{bmatrix}
\in \mathbb{R}^{35 \times 2}
$$

또한 dataset의 'Salary' column을 아래와 같이 하나의 column vector로 설정하자.

$$
\mathbf{y} =
\begin{bmatrix}
y^{(1)} & y^{(2)} & \cdots & y^{(35)}
\end{bmatrix}^T
\in \mathbb{R}^{35}
$$

## 2. Notations

본격적으로 문제를 해결하기 전, 표기를 확립하자.

| Notation | Meanings |
| --- | --- |
| $m$ | example의 개수 (이 문제에서 $m=35$) |
| $n$ | feature의 개수 (이 문제에서 $n=1$) |
| $x^{(i)} \in \mathbb{R}$ | input of example $i$ |
| $y^{(i)} \in \mathbb{R}$ | target value of example $i$ |
| $\hat{y}^{(i)} \in \mathbb{R}$ | example $i$에 대한 model의 prediction |
| $X \in \mathbb{R}^{m \times (n+1)}$ | design matrix $(X_{i,:}=\begin{bmatrix}1 & x_1^{(i)} & \cdots & x_n^{(i)}\end{bmatrix})$ |
| $\mathbf{y}=\begin{bmatrix}y^{(1)} & \cdots & y^{(m)}\end{bmatrix}^T \in \mathbb{R}^{m}$ | target vector |
| $\mathbf{\hat{y}}=\begin{bmatrix}\hat{y}^{(1)} & \cdots & \hat{y}^{(m)}\end{bmatrix}^T \in \mathbb{R}^{m}$ | prediction vector |
| $\mathbf{\theta}=\begin{bmatrix}\theta_0 & \theta_1 & \cdots & \theta_n \end{bmatrix}^T \in \mathbb{R}^{n+1}$ | parameter |
| $h_\theta(x)$ | hypothesis function $(y^{(i)}=h_\theta(x^{(i)}))$ |
| $J(\theta)$ | loss function |

## 3. Batch Gradient Descent

Hypothesis function을 parameter에 대해 다음과 같이 설정하자.

$$
h_{\theta}(x^{(i)})=\theta_0x_0^{(i)}+\theta_1x_1^{(i)}
$$

위와 같이 설정한 이유는 이후 loss function의 gradient를 구할 때 수식 표현의 편리성을 위해서이다 $(\begin{bmatrix}x_0^{(i)}&x_1^{(i)}\end{bmatrix}=\begin{bmatrix}1&x^{(i)}\end{bmatrix})$.

Loss function을 다음과 같이 half-MSE로 설정하자.

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)}-\hat{y}^{(i)})^2
$$

이때 $\hat{y}^{(i)}$는 다음과 같이 나타낼 수 있다.

$$
h_{\theta}(x^{(i)})=X_{i,:}\mathbf{\theta}
$$

우리는 아래와 같은 optimization problem을 풀기 위해 gradient descent algorithm을 사용한다.

$$
\theta^{\ast}=\underset{\theta}{\arg\min} J(\theta)
$$

Gradient descent algorithm은 다음과 같이 진행된다.

1. Initialize parameter: $\theta = \theta_{init}$
2. Update parameter until convergence: $\theta \leftarrow \theta - \alpha \nabla_{\theta} J(\theta)$

1번에서 우리는 $\theta_{init}$을 zero vector, $\begin{bmatrix}0 & 0\end{bmatrix}^T$로 설정한다.

아래에서는 2번 과정을 자세히 서술한다.

먼저 learning rate $\alpha$는 관습에 따라 0.01로 놓는다. 아래는 loss function의 parameter $\theta$의 $j$번째 성분$(j=0,1)$에 대한 gradient를 구하는 과정이다.

$$
\begin{align}
\frac{\partial J}{\partial\theta_j} &=
\frac{\partial}{\partial\theta_j} (\frac{1}{2m} \sum_{i=1}^{m} (y^{(i)}-\hat{y}^{(i)})^2) \\ &=
\frac{\partial}{\partial\theta_j} (\frac{1}{2m} \sum_{i=1}^{m} (y^{(i)}-(\theta_0x_0^{(i)}+\theta_1x_1^{(i)}))^2) \\ &=
\frac{1}{m} \sum_{i=1}^{m} (y^{(i)}-(\theta_0x_0^{(i)}+\theta_1x_1^{(i)}))(-x_j^{(i)}) \\ &=
\frac{1}{m}X_{:,j}^T(\mathbf{\hat{y}}-\mathbf{y})
\end{align}
$$

따라서 $\nabla_{\theta}J(\theta)=\frac{1}{m}X^T(\mathbf{\hat{y}}-\mathbf{y})$이다.

2번 과정을 learning rate $\alpha=0.01$에 대해 10000번 수행하면 다음과 같이 업데이트된 parameter를 얻는다.

$$
\theta_0=28859.526193,\,\theta_1=8731.941047
$$

즉, 업데이트된 parameter를 반영한 hypothesis function은 아래와 같다.

$$
h_{\theta}(x) = 28859.526193 + 8731.941047x
$$

아래는 2번 과정에서 10번째 iteration까지 parameter가 업데이트됨에 따라 fitting된 hypothesis function을 보여준다.

![Linear Regression Fitting Process](/assets/img/2026-10-25/line_fitting.gif)

## 4. Least Squares Method

Gradient descent는 gradient를 반복적으로 계산하여 parameter를 업데이트하는 방법이다.
반면 least squares method에서는 target value와 prediction과의 차이의 제곱합을 최소화하는 문제를 normal equation으로 표현하고, 해당 linear system을 풀어 optimal parameter를 구할 수 있다.
Normal equation은 다음과 같다.

$$
X^TX\theta=X^T\mathbf{y}
$$

Normal equation의 solution은 다음과 같다.

$$
\theta_0=28859.526195,\,\theta_1=8731.941047
$$

두 방법으로 구한 parameter의 차이를 L2 norm과 relative error로 비교한 결과, 두 값이 매우 가까움을 확인했다. 이는 gradient descent가 normal equation으로 구한 solution에 근접했음을 보여준다.

## Reference
- Data Source: [Kaggle - Salary Dataset (by Rashid Sadiq)](https://www.kaggle.com/datasets/rsadiq/salary)
