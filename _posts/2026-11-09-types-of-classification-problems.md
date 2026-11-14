---
title: "ML Theory - Types of Classification Problems"
date: 2026-11-09
categories: [Machine Learning, Theory]
tags: [AI, ML, supervised-learning, classification, binary, multi-class, multi-label]
math: true
---

## Types of Classification Problems

Machine learning 뿐만 아니라 모든 문제 상황에서 가장 먼저 해야 하는 것은, 문제를 이해하고 목적을 찾는 것이다.

주어진 dataset 혹은 풀어야 하는 task를 분석하여 해결해야 하는 문제가 무엇인지 파악하는 것은 model 선택 등에 큰 도움을 줄 것이다.

Machine learning을 공부하다 보면, 'multi-class', 'multi-label' 등의 용어가 자주 등장하고, 이는 꽤 헷갈릴 만하다.

이 포스트를 통해 해당 개념들을 확실히 숙지하여 더는 헷갈릴 일이 없길 바란다.

## 1. Classification Problems

Classification problem은 supervised learning의 한 종류로, 주어진 dataset에서 target value가 categorical data인 경우이다. 즉, 사람의 키를 통해 몸무게를 예측하듯이 continuous한 numerical data를 예측하는 게 아니라, 어떤 동물의 이미지가 고양이 사진인지 강아지 사진인지를 구분하듯이 discrete한 categorical data를 예측하는 것이다.

## 2. Difference between 'Class' and 'Label'

### 2-1. Class

Class는 classification 문제에서 data가 속하는 category를 뜻한다.

어떤 동물 사진이 주어졌을 때 고양이 사진인지 아닌지를 구분하는 classification 문제를 생각해보자. 예를 들어 Dataset에 속한 각 sample은 label에 의해 '고양이임' 혹은 '고양이가 아님' 둘 중 하나로 구분될 것이다. 이 문제는 '고양이임', '고양이가 아님'의 두 가지 class를 갖는 것이다.

다른 예로 사람의 혈액형을 구분하는 문제를 생각해보면, 이 문제는 'A형', 'B형' 'AB형', 그리고 'O형'의 총 4가지 class를 갖는다.

### 2-2. Label

Label은 dataset sample에 부여된 target value를 뜻한다.

동물 사진을 feature로 갖고 각 sample에 '고양이임' 혹은 '고양이가 아님' 둘 중 하나가 부여된 dataset을 가정하자. 이때 각 sample에 부여된 '고양이임' 혹은 '고양이가 아님'을 각 sample의 label이라고 한다. 이 문제의 경우(하나의 사진에 동물이 한 마리만 있다고 가정하면) 하나의 사진이 고양이 사진이 아니면서 동시에 고양이 사진일 수는 없다. 두 class가 서로 exclusive하기 때문이다.

헷갈리는 지점은 이제부터다.

영화의 제목을 통해 장르를 예측하는 문제를 생각해보자. 어떤 영화는 SF이면서 스릴러이거나, 코미디이면서 로맨스일 수 있다. 각 class, SF, 스릴러, 코미디, 그리고 로맨스는 서로 exclusive하지 않기 때문에 하나의 영화(하나의 sample)는 여러 class로 label될 수 있다.

## 3. Types of Classification Problems

### 3-1. Binary Classification Problem

Binary classification problem은 sample이 가질 수 있는 class가 2개이며 이 중 하나에 속한다고 예측하는 문제이다.

즉, single-label two-class classification problem인 것이다.

Dataset에서 각 target value $y^{(i)}$가 취할 수 있는 값을 수식으로 표현하면, 아래와 같다.

$$
y^{(i)} \in \set{0, 1}
$$

이때 label $0$과 $1$은 각각 negative, positive class이다. 예를 들어 고양이 사진이 아니면 $0$, 맞다면 $1$이라고 label하는 것이다.

### 3-2. Multi-class classification Problem

Multi-class classification이란 표현은 일반적으로 single-label multi-class classification problem을 뜻한다. 각 sample은 하나의 class에만 속할 수 있다는 뜻이다 (각 class는 서로 exclusive하며 하나의 sample이 각 class에 속할 확률의 합은 1이다).

3 이상인 $K$개의 class를 class $1,2,\ldots,K$라 하면, 각 target value는 아래와 같이 값을 취할 수 있다.

$$
y^{(i)} \in \set{1,2,\ldots,K}
$$

이때 각 class에 매긴 정수값을 수치 데이터처럼 취급하면, 실제로 존재하지 않는 순서나 크기 관계가 model에 반영될 수 있다.

Dataset의 모든 sample은 고양이 사진, 강아지 사진, 햄스터 사진 셋 중 하나이다. 이외의 동물은 고려하지 않는다. 이때 고양이 class를 1, 강아지 class를 2, 햄스터 class를 3이라고 하면

$$
\text{cat} + \text{cat} = \text{dog}, \quad \text{cat} + \text{dog} = \text{hamster}
$$

와 같은 의도치 않은 결과가 발생할 수 있다.

따라서 각 class를 $1$부터 $K$까지 넘버링하는 방법이 안 좋을 수 있다. 위와 같은 일을 막기 위해 **one-hot encoding** 등의 기법이 쓰인다. One-hot encoding이란, 각 sample의 label을 class의 개수만큼 성분을 갖고, sample이 속하는 class에 대응되는 성분만 1, 나머지 성분은 0으로 채워진 vector로 표현하는 방법이다.

고양이/강아지/햄스터 문제로 예를 들면,

$$
\mathbf{y}^{(i)} =
\begin{cases}
    \begin{bmatrix}
        1 & 0 & 0
    \end{bmatrix},
    \quad y^{(i)}_\text{class} = \text{cat} \\
    \begin{bmatrix}
        0 & 1 & 0
    \end{bmatrix},
    \quad y^{(i)}_\text{class} = \text{dog} \\
    \begin{bmatrix}
        0 & 0 & 1
    \end{bmatrix},
    \quad y^{(i)}_\text{class} = \text{hamster} \\
\end{cases}
$$

와 같이 각 column을 고양이, 강아지, 그리고 햄스터인지 여부를 $0$ 또는 $1$로 나타내게끔 원본 data를 one-hot vector로 변환하는 것이다.

One-hot encoding이 필수라는 뜻은 아니다. Data를 표현하는 방법 중 하나이다.

### 3-3. Multi-lable Classification Problem

Multi-label classification problem은 하나의 sample에 두 개 이상의 class 혹은 tag가 label될 수 있는 문제이다. Class가 총 $K$개 있다고 가정하면 각 target value는 아래와 같이 값을 취할 수 있다.

$$
\mathbf{y}^{(i)} \in \set{0,1}^K
$$

이 표현이 multi-class classification 문제의 one-hot encoding과 다른 점은, 두 개 이상의 성분이 1일 수 있다는 점이다. 'Multi-label' 말 그대로 한 sample이 여러 class에 label될 수 있다.

영화의 장르 class가 다음과 같이 총 K개 있다고 하자.

$$
\text{SF, Romance, Horror, $\ldots$, Comedy}
$$

어떤 영화의 장르가 Romance이면서 Comedy이고, 나머지 장르에는 속하지 않을 때 그 영화는 다음과 같이 표현할 수 있다.

$$
\begin{bmatrix}
    0 & 1 & 0 & 0 & \cdots & 0 & 0 & 1
\end{bmatrix}
$$ 

### 3-4. Multi-Output Classification Problem

Multi-Output classification problem은 여러 개의 독립적인 target variable을 동시에 예측하는 문제이다.

한 동물 객체가 담긴 사진을 통해

1. 동물은 고양이/강아지/햄스터 중 무엇인가?
2. 동물은 암컷인가, 수컷인가?

두 물음에 답하는 것이다.

위의 1번과 2번은 각각 multiclass classification problem이다 (고양이/강아지/햄스터 class는 서로 exclusive, 암컷/수컷은 서로 exclusive). 즉, 이 문제는 종과 성별 두 가지 target을 가지며 각 target은 class를 3개, 2개를 갖는 multi-output classification problem이다.

$N$개의 target을 갖고, 각 target $i$가 $K_i$개의 class를 가질 때, target value는 다음과 같이 값을 취할 수 있다.

$$
\mathbf{y}^{(i)} \in \set{1,\ldots,K_1} \times \cdots \times \set{1,\ldots,K_N}
$$

One-hot vector로 표현하면,

$$
\mathbf{y}^{(i)} =
\begin{bmatrix}
    \mathbf{y}_1^{(i)} \mid \mathbf{y}_2^{(i)} \mid \cdots \mid \mathbf{y}_N^{(i)}
\end{bmatrix}
$$

단,

$$
\mathbf{y}_j^{(i)} \in \set{0,1}^{K_j}, \quad \sum_{k=1}^{K_j}(y_j^{(i)})_k=1
$$