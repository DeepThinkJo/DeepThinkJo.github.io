---
title: "Prerequisite Mathematics - Least Squares Problem"
date: 2026-10-29
categories: [Prerequisite Mathematics, Linear Algebra]
tags: [mathematics, linear-algebra]
math: true
---

## Least Squares Problem

Least squares problem과 이를 해결하기 위한 normal equation을 소개하는 포스트이다.
Least squares problem이 무엇인지, 왜 필요한지, 그리고 이 문제를 풀기 위해 normal equation을 유도하는 과정을 담고 있다.

## 1. Motivation & Defining Problem

Least squares problem이 필요한 경우를 이해하려면 linear system의 solution과 관련된 몇 가지 개념을 이해해야 한다.

어떤 linear system

$$
A\mathbf{x}=\mathbf{b}
$$

에 대해 equation이 성립하게 하는 $\mathbf{x}$를 linear system의 **solution**이라 한다.
이때 주어진 linear system이 solution이 있다면 **consistent**하다고 하고, 아니라면 **inconsistent**하다고 한다.

또한 solution이 존재하는 것은 $\mathbf{b} \in \operatorname{Col}(A)$와 동치이다.

Least squares problem은 특히 linear system $A\mathbf{x}=\mathbf{b}$이 inconsistent한 경우에 주목한다.

주어진 linear system의 exact solution이 존재하지 않더라도, 두 vector $\mathbf{b}$와 $A\mathbf{x}$ 사이의 squared Euclidean distance를 최소화하는 vector $\hat{\mathbf{x}}$를 찾는 것이다.

$$
\hat{\mathbf{x}} \in \underset{\mathbf{x}}{\arg\min} \mid\mid \mathbf{b} - A\mathbf{x} \mid\mid_2^2
$$

## 2. Notations

본격적으로 least squares problem과 normal equation을 다루기 전, 표기를 확립하자.

|Notation|Meaning|
|---|---|
|$A \in \mathbb{R}^{m \times n}$|coefficient matrix|
|$\mathbf{b} \in \mathbb{R}^m$|output column vector|
|$\mathbf{x} \in \mathbb{R}^n$|unknown vector|
|$\hat{\mathbf{x}} \in \mathbb{R}^n$|least squares solution|
|$\mathbf{e} = \mathbf{b}-A\mathbf{x} \in \mathbb{R}^m$|residual vector|
|$\hat{\mathbf{e}}=\mathbf{b}-A\hat{\mathbf{x}} \in \mathbb{R}^m$|optimal residual|
|$A_{i,j} \in \mathbb{R}$|element on $i$th row and $j$th column|
|$A_{i,:} \in \mathbb{R}^{1 \times n}$|$i$th row of matrix $A$|
|$A_{:,j} \in \mathbb{R}^{m \times 1}$|$j$th column of matrix $A$|
|$\operatorname{Col}(A)=\set{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^{n}}$|column space of $A$|

## 3. Deriving Normal Equation Using Orthogonality Concepts

Linear algebra의 orthogonality 관련 개념을 활용하면 least squares problem에서 normal equation을 유도해낼 수 있다.

### 3-1. Residual Vector is Orthogonal to the Column Space

핵심은 아래와 같다.

$$
(\mathbf{b}-A\hat{\mathbf{x}}) \in \operatorname{Col}(A)^\perp
\Leftrightarrow \,
\hat{\mathbf{x}} \in \underset{\mathbf{x}}{\arg\min} \mid\mid \mathbf{b} - A\mathbf{x} \mid\mid_2^2
$$

즉, $\mathbf{b}-A\hat{\mathbf{x}}$가 $\operatorname{Col}(A)$와 직교하는 것은 $\hat{\mathbf{x}}$가 least squares problem의 optimal solution인 것과 필요충분조건이다.

(Proof of the Forward Direction) 다음을 보이면 충분하다.

$$
\forall \mathbf{x} \in \mathbb{R}^n, \, \mid\mid \mathbf{b} - A\mathbf{x} \mid\mid_2^2 \, \ge \, \mid\mid \mathbf{b} - A\hat{\mathbf{x}} \mid\mid_2^2
$$

임의의 $\mathbf{x} \in \mathbb{R}^n$에 대해, 다음과 같이 vector $\mathbf{v}$를 정의하자.

$$
\mathbf{v} = A\mathbf{x}-A\hat{\mathbf{x}} = A(\mathbf{x}-\hat{\mathbf{x}}) \in \mathbb{R}^m
$$

이때 $\mathbf{v}$는 $A$의 column vector들의 linear combination이므로 가정에 의해 optimal residual vector인 $\hat{\mathbf{e}}=\mathbf{b}-A\hat{\mathbf{x}}$와 직교한다 $(\mathbf{v} \perp \hat{\mathbf{e}})$.

$$
\begin{align}
\mathbf{b} - A\mathbf{x} &=
\mathbf{b} - (A\mathbf{x} - A\hat{\mathbf{x}}) - A\hat{\mathbf{x}} \\ &=
(\mathbf{b}-A\hat{\mathbf{x}}) - (A\mathbf{x} - A\hat{\mathbf{x}}) \\ &=
\hat{\mathbf{e}} - \mathbf{v}
\end{align}
$$

이때 $\hat{\mathbf{e}}$와 $\mathbf{v}$는 직교하므로,

$$
\begin{align}
    \mid\mid \mathbf{b} - A\mathbf{x} \mid\mid_2^2 &=
    \mid\mid \hat{\mathbf{e}} - \mathbf{v} \mid\mid_2^2 \\ &=
    \mid\mid \hat{\mathbf{e}} \mid\mid_2^2 + \mid\mid \mathbf{v} \mid\mid_2^2 \\ &\ge
    \mid\mid \hat{\mathbf{e}} \mid\mid_2^2 \\ &=
    \mid\mid \mathbf{b}-A\hat{\mathbf{x}} \mid\mid_2^2

\end{align}
$$

(Proof of the Reverse Direction) 임의의 vector $\mathbf{v} \in \operatorname{Col}(A)$에 대해 $A\mathbf{z}=\mathbf{v}$를 만족하는 vector $\mathbf{z} \in \mathbb{R}^n$을 설정하자.

다음과 같이 $t \in \mathbb{R}$에 대한 함수를 정의하자.

$$
\mathbf{x}(t) = \hat{\mathbf{x}}+t\mathbf{z}
$$

$\hat{\mathbf{x}}$가 least squares problem의 solution이므로 다음이 성립한다.

$$
\forall t \in \mathbb{R}, \,
\mid\mid \mathbf{b} - A\mathbf{x}(0) \mid\mid_2^2 \, \le \,
\mid\mid \mathbf{b} - A\mathbf{x}(t) \mid\mid_2^2
$$

위 조건을 활용하여 $\hat{\mathbf{e}}^T\mathbf{v}=0$을 보이면 충분하다. 왜냐하면 $\mathbf{v}$는 $A$의 column space 내의 임의의 vector이므로 이는 $\hat{\mathbf{e}}$가 $A$의 column space와 직교함과 동치이기 때문이다.

위 부등식의 좌변은 다음과 같다.

$$
\begin{align}
    \mid\mid \mathbf{b} - A\mathbf{x}(0) \mid\mid_2^2 &=
    \mid\mid \mathbf{b} - A\hat{\mathbf{x}} \mid\mid_2^2 \\ &=
    \mid\mid \hat{\mathbf{e}} \mid\mid_2^2
\end{align}
$$

우변은 다음과 같다.

$$
\begin{align}
    \mid\mid \mathbf{b} - A\mathbf{x}(t) \mid\mid_2^2 &=
    \mid\mid \mathbf{b} - A(\hat{\mathbf{x}}+t\mathbf{z}) \mid\mid_2^2 \\ &=
    \mid\mid \mathbf{b} - A\hat{\mathbf{x}}-tA\mathbf{z} \mid\mid_2^2 \\ &=
    \mid\mid \hat{\mathbf{e}} - t\mathbf{v} \mid\mid_2^2 \\ &=
    \mid\mid \hat{\mathbf{e}} \mid\mid_2^2 - 2t\hat{\mathbf{e}}^T\mathbf{v} + t^2\mid\mid \mathbf{v} \mid\mid_2^2
\end{align}
$$

각각을 부등식에 대입하여 정리하면 다음과 같은 $t$에 대한 부등식이 도출된다.

$$
\forall t \in \mathbb{R}, \,
(\mid\mid \mathbf{v} \mid\mid_2^2)t^2 - 2(\hat{\mathbf{e}}^T\mathbf{v})t \ge 0
$$

이때 $\mid\mid \mathbf{v} \mid\mid_2^2=0$인 경우, $\hat{\mathbf{e}}^T\mathbf{v}=0$는 자명하다.
따라서 $\mid\mid \mathbf{v} \mid\mid_2^2 > 0$로 $t$에 대한 이차부등식인 경우를 생각해보자.
모든 실수 $t$에 대해 성립해야 하므로, $f(t)=(\mid\mid \mathbf{v} \mid\mid_2^2)t^2 - 2(\hat{\mathbf{e}}^T\mathbf{v})t$와 같이 함수 $f$를 설정하면, 이차방정식 $f(t)=0$의 판별식은 $0$보다 작거나 같아야 한다.

$$
\begin{align}
    D &=
    (-2\hat{\mathbf{e}}^T\mathbf{v})^2 - 4\mid\mid \mathbf{v} \mid\mid_2^2 \cdot 0 \\ &=
    4(\hat{\mathbf{e}}^T\mathbf{v})^2 \\ &\le
    0
\end{align}
$$

따라서 $\hat{\mathbf{e}}^T\mathbf{v}=0$이 도출된다.

### 3-2. Deriving Normal Equation

위에서 증명한 성질을 바탕으로 normal equation을 도출해보자.

$\mathbf{e}$가 $\operatorname{Col}(A)$에 직교한다는 것은, $\operatorname{Col}(A)$ 내의 각 vector와 $\mathbf{e}$의 dot product가 0이라는 뜻이다. 이는 다음과 동치이다.

$$
A_{:,j}^T\hat{\mathbf{e}}=0, \, \text{for} \, j=1, \ldots, n
$$

또한 matrix operation으로 표현하면 다음과도 동치이다.

$$
A^T\hat{\mathbf{e}}=\mathbf{0}
$$

위 식에 $\hat{\mathbf{e}}=\mathbf{b}-A\hat{\mathbf{x}}$를 대입하면 normal equation을 얻을 수 있다.

$$
\begin{align}
    A^T\hat{\mathbf{e}} &=
    A^T(\mathbf{b} - A\hat{\mathbf{x}}) \\ &=
    A^T\mathbf{b} - A^TA\hat{\mathbf{x}} \\ &=
    \mathbf{0}
\end{align}
$$

정리하면,

$$
A^TA\hat{\mathbf{x}}=A^T\mathbf{b}
$$

즉, 위 normal equation의 solution이 주어진 least squares problem의 solution이다.

## 4. Deriving Normal Equation Using Gradient Concepts

이번에는 optimization 관점에서 normal equation을 유도해보자.

일반적인 optimization problem

$$
\hat{x} = \underset{x}{\arg\min} J(x)
$$

에서 $\nabla_x J(\hat{x})=0$인 조건은 $\hat{x}$가 local optimum이기 위한 First-Order Necessary Condition (FONC)이다. 즉, 이 조건을 만족하더라도 local optimum이 아닐 수 있다.

하지만 objective function이 convex function이라면, FONC는 global optimum을 위한 충분조건이 된다.
Least squares problem의 경우 objective function이 convex function이므로 FONC를 만족하는 $\hat{\mathbf{x}}$는 global optimum이다.

아래에서 이를 증명한다.

Least squares problem은 다음을 만족하는 $\hat{\mathbf{x}}$를 찾는 것이다.

$$
\hat{\mathbf{x}} = \underset{\mathbf{x}}{\arg\min}
\mid\mid \mathbf{b} - A\mathbf{x} \mid\mid_2^2
$$

따라서 objective function $J(\mathbf{x})$는 다음과 같다.

$$
\begin{align}
    J(\mathbf{x}) &=
    \mid\mid \mathbf{b} - A\mathbf{x} \mid\mid_2^2 \\ &=
    (\mathbf{b} - A\mathbf{x})^T(\mathbf{b} - A\mathbf{x})
\end{align}
$$

Objective function의 $\mathbf{x}$에 대한 gradient를 구해보자.

$$
\begin{align}
    \nabla_\mathbf{x} J(\mathbf{x}) &=
    \nabla_\mathbf{x} (\mathbf{b} - A\mathbf{x})^T(\mathbf{b} - A\mathbf{x}) \\ &=
    \nabla_\mathbf{x} (\mathbf{b}^T\mathbf{b} - \mathbf{b}^TA\mathbf{x} - \mathbf{x}^TA^T\mathbf{b} + \mathbf{x}^TA^TA\mathbf{x}) \\ &=
    2A^TA\mathbf{x} - 2A^T\mathbf{b}
\end{align}
$$

$\nabla_\mathbf{x} J(\mathbf{x})=0$을 만족하는 $\mathbf{x}$가 FONC를 만족하는 stationary point이다.
이때 그 stationary point는 normal equation의 solution임을 기억해두자.

이제 objective function의 Hessian을 구해보자.

$$
\begin{align}
    \nabla_\mathbf{x}^2 J(\mathbf{x}) &=
    \nabla_\mathbf{x} (2A^TA\mathbf{x} - 2A^T\mathbf{b}) \\ &=
    2A^TA
\end{align}
$$

임의의 vector $\mathbf{z} \in \mathbb{R}^n$에 대해,

$$
\begin{align}
    \mathbf{z}^T (\nabla_\mathbf{x}^2 J(\mathbf{x})) \mathbf{z} &=
    \mathbf{z}^T (2A^TA) \mathbf{z} \\ &=
    2\mathbf{z}^TA^TA\mathbf{z} \\ &=
    2 (A\mathbf{z})^T(A\mathbf{z}) \\ &=
    2 \mid\mid A\mathbf{z} \mid\mid_2^2 \\ &\ge
    0
\end{align}
$$

이므로, Hessian은 positive semidefinite matrix이다.
따라서 objective function은 convex function이다.

결론짓자면, least squares problem에서 objective function이 convex function이므로 objective function에 대해 FONC를 만족하는 stationary point는 global minimum이다.

또한 FONC를 만족하는 stationary point는 normal equation의 solution과 같다.

## 5. Existence and Uniqueness of Least Squares Solution

위의 섹션들은 normal equation $A^TA\mathbf{x}=A^T\mathbf{b}$를 만족하는 solution이 least squares problem의 global solution임을 정당화한다.

이제 그 solution의 existence와 uniqueness를 논해보자.

### 5-1. Existence

주어진 least squares problem의 solution은 항상 존재한다. 즉,

$$
\forall \mathbf{b} \in \mathbb{R}^m, \,
\exists \mathbf{x} \, \text{such that} \, A^TA\mathbf{x} = A^T\mathbf{b}
$$

이는 임의의 vector $\mathbf{b} \in \mathbb{R}^m$에 대해 $A^T\mathbf{b}$ 항상 $\operatorname{Col}(A^TA)$의 원소라는 뜻이다.

이를 보이려면 먼저 $\operatorname{Col}(A^TA)=\operatorname{Col}(A^T)$임을 보여야 한다. 이때 $A^T\mathbf{b} \in \operatorname{Col}(A^T)$이므로, 두 column space가 같다는 사실을 이용하면 $A^T\mathbf{b} \in \operatorname{Col}(A^TA)$임을 알 수 있다.

$\operatorname{Col}(A^TA)$ 내의 임의의 vector $\mathbf{y}$에 대해 $\mathbf{y}=A^TA\mathbf{x}$를 만족하는 vector $\mathbf{x}$가 존재한다.
이때 $\mathbf{y}=A^T(A\mathbf{x})$이므로 $\mathbf{y}$는 $\operatorname{Col}(A^T)$의 원소이다.
따라서 $\operatorname{Col}(A^TA) \subseteq \operatorname{Col}(A^T)$이다.

이때 $\operatorname{nullity}(A^TA)=\operatorname{nullity}(A)$를 활용한다.

$$
\begin{align}
    \operatorname{rank}(A^TA) &=
    n - \operatorname{nullity}(A^TA) \\ &=
    n - \operatorname{nullity}(A) \\ &=
    \operatorname{rank}(A) \\ &=
    \operatorname{rank}(A^T)
\end{align}
$$

$\operatorname{Col}(A^T)$와 그것의 subspace인 $\operatorname{Col}(A^TA)$가 rank가 같으므로 두 vector space는 같다.

이때 $A^T\mathbf{b} \in \operatorname{Col}(A^T)$인데, $\operatorname{Col}(A^TA)=\operatorname{Col}(A^T)$이므로 $A^T\mathbf{b} \in \operatorname{Col}(A^TA)$이다.

따라서 모든 $\mathbf{b} \in \mathbb{R}^m$에 대해 normal equation $A^TA\mathbf{x} = A^T\mathbf{b}$의 solution은 항상 존재한다.

### 5-2. Uniqueness

Solution의 uniqueness는 $A$의 rank와 관련있다.

결론부터 말하자면, $\operatorname{rank}(A)=n$이면 unique solution을 갖고, 그렇지 않다면 solution은 유일하지 않다.

Least squares problem에 대응되는 normal equation은 다음과 같다.

$$
A^TA\mathbf{x} = A^T\mathbf{b}
$$

이때 $A^TA$가 inverse matrix를 갖는다면, 위 normal equation의 solution은 아래와 같이 유일하게 결정된다.

$$
\mathbf{x} = (A^TA)^{-1}A^T\mathbf{b}
$$

$A^TA$가 invertible할 조건은, $\operatorname{rank}(A^TA)=n$이다.
하지만

$$
\begin{align}
    \operatorname{rank}(A^TA) &=
    n - \operatorname{nullity}(A^TA) \\ &=
    n - \operatorname{nullity}(A) \\ &=
    \operatorname{rank}(A)
\end{align}
$$

이므로, $\operatorname{rank}(A)=n$이면 $A^TA$는 invertible하다.

즉, $A$가 full column rank를 갖는다면 normal equation의 solution은 unique하다.

이는 4번 섹션에서 언급한 objective function의 Hessian과도 연결지어 설명할 수 있다.
Least squares problem의 objective function에 대한 Hessian은 다음과 같다.

$$
\nabla_\mathbf{x}^2 J(\mathbf{x}) = 2A^TA
$$

만약 $\operatorname{rank}(A)=n$이라면 Hessian인 $2A^TA$는 positive definite matrix이고, 결국 objective function은 strictly convex function이 되는 것이다.

일반적으로 $A$가 full column rank matrix가 아닌 경우에는, normal equation의 solution이 unique하지 않다. 실전에서는 Moore-Penrose pseudoinverse $A^+$를 활용하여 minimum-norm least squares solution을 구할 수 있다.