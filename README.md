# Attention Mechanisms in Sequence-to-Sequence Models

## Overview

The main objective of this project is to understand scaled dot-product attention and to implement a simple attention mechanism in a sequence-to-sequence model. The task involves translating sentences from German to English using the Multi30k dataset, which contains over 31,000 bitext sentences describing common visual scenes in both languages.

### 1. Scaled Dot-Product Attention

Starts from the definition of a single-query scaled dot-product attention mechanism. Given a query $\mathbf{q} \in \mathbb{R}^{1\times d}$, a set of candidates represented by keys $\mathbf{k}_1, ... , \mathbf{k}_m \in \mathbb{R}^{1\times d}$ and values $\mathbf{v}_1, ... , \mathbf{v}_m \in \mathbb{R}^{1\times d_v}$, we compute the scaled dot-product attention as:

$$
\alpha_i = \frac{\mbox{exp}\left(~\mathbf{q}\mathbf{k}_ i^T / \sqrt d\right)}{\sum_{j=1}^m \mbox{exp}\left(\mathbf{q}\mathbf{k}_ j^T / \sqrt d\right)}
$$

$$
\textbf{a} = \sum_{j=1}^m \alpha_j \mathbf{v}_j
$$

where the $\alpha_i$ are referred to as attention values (or collectively as an attention distribution).

#### **1.1 Copying**
Q. Describe what properties of the keys and queries would result in the output $\textbf{a}$ being equal to one of the input values $\mathbf{v}_j$. Specifically, what must be true about the query $\mathbf{q}$ and the keys $\mathbf{k}_1, ..., \mathbf{k}_m$ such that $\textbf{a} \approx \mathbf{v}_j$? (We assume all values are unique -- $\mathbf{v}_i \neq \mathbf{v}_j,~\forall ij$.)

> In the case where $\textbf{a} \approx \mathbf{v}_j$, the similarity score for $\mathbf{q}\mathbf{k}_j^T$ is significantly higher than all others due to the softmax function producing outputs to have probability distribution. Therefore, given a query $\mathbf{q}$, $\mathbf{k}_j$ must be significantly higher than others to determine the similarity score.

#### **1.2 Average of Two**
Q. Consider a set of key vectors $\mathbf{k}_1, ... , \mathbf{k}_m$ where all keys are orthogonal unit vectors -- that is to say $\mathbf{k}_i \mathbf{k}_j^T = 0, \forall ij$ and $\Vert\mathbf{k}_i\Vert=1,\forall i$. Let $\mathbf{v}_a, \mathbf{v}_b \in \{\mathbf{v}_1, ..., \mathbf{v}_m\}$ be two value vectors. Give an expression for a query vector $\mathbf{q}$ such that the output $\textbf{a}$ is approximately equal to the average of $\mathbf{v}_a$ and $\mathbf{v}_b$, that is to say $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$. You can reference the key vectors corresponding to $\mathbf{v}_a$ and $\mathbf{v}_b$ as $\mathbf{k}_a$ and $\mathbf{k_b}$ respectively.

>From $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$, we can consider the term $\frac{1}{2}$ is from $\alpha_i$. Meaning that $\alpha_a = \alpha_b$ and $\alpha_i = 0$ should be satisfied to meet the condition. Since $\alpha_i = \mbox{softmax}(\mathbf{q}\mathbf{k}_i^T)$, we only want to keep $\mathbf{k}_a$ and $\mathbf{k}_b$; otherwise, $\mathbf{k}_i=0$.\
Considering $c$ as a large constant, the below expression for $\mathbf{q}$ can satisfy $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$.
>
>$$ \mathbf{q}=c(\mathbf{k}_a + \mathbf{k}_b) $$
>
>By constructing the original linear equation of $\mathbf{q}$ and $\mathbf{k}$, we can ensure if this expression satisfy the condition.
>
>$$
\mathbf{q}\mathbf{k}_a^T=c(\mathbf{k}_a\mathbf{k}_a^T + \mathbf{k}_b\mathbf{k}_a^T)=c(1+0)=c
$$
>
>$$
\mathbf{q}\mathbf{k}_b^T=c(\mathbf{k}_a\mathbf{k}_b^T + \mathbf{k}_b\mathbf{k}_b^T)=c(0+1)=c
$$
>
>$$
\mathbf{q}\mathbf{k}_i^T=c(\mathbf{k}_a\mathbf{k}_i^T + \mathbf{k}_b\mathbf{k}_i^T)=c(0+0)=0
$$
>
>Therefore, $\alpha_a = \alpha_b = c$ and $\alpha_i = 0$. $\textbf{a}$ can be written,
>
>$$
\textbf{a} = \alpha_a \mathbf{v}_ a + \alpha_b \mathbf{v}_ b + \sum_{j=1, j\neq a, b}^m \alpha_j \mathbf{v}_j
$$
>
>$$
\textbf{a} \approx \frac{1}{2}\mathbf{v}_a + \frac{1}{2}\mathbf{v}_b + 0
$$
>
>$$
\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)
$$

#### **1.3 Noisy Average**
Q. Now consider a set of key vectors $\{\mathbf{k}_1, ... , \mathbf{k}_m\}$ where keys are randomly scaled such that $\mathbf{k}_i = \mathbf{\mu}_i*\lambda_i$ where $\lambda_i \sim \mathcal{N}(1, \beta)$ is a randomly sampled scalar multiplier. Assume the unscaled vectors $\mu_1, ..., \mu_m$ are orthogonal unit vectors. If you use the same strategy to construct the query $q$ as you did in Task 1.2, what would be the outcome here? Specifically, derive $\mathbf{q}\mathbf{k}_a^T$ and $\mathbf{q}\mathbf{k}_b^T$ in terms of $\mu$'s and $\lambda$'s. Qualitatively describe what how the output $a$ would vary over multiple resamplings of $\lambda_1, ..., \lambda_m$.

>From the expression for $\mathbf{q}$ in Task 1.2,
>
>$$
\mathbf{q}=c(\mathbf{k}_a + \mathbf{k}_b)
$$
>
>By substituting $\mathbf{k}_i = \mathbf{\mu}_i*\lambda_i$,
>
>$$
\mathbf{q}=c(\mathbf{\mu}_a*\lambda_a + \mathbf{\mu}_b*\lambda_b)
$$
>
>The expression for $\mathbf{q}\mathbf{k}_a^T$ and $\mathbf{q}\mathbf{k}_b^T$,
>
>$$
\mathbf{q}\mathbf{k}_a^T=c(\lambda_a^2*\mathbf{\mu}_a\mathbf{\mu}_a^T + \lambda_a\lambda_b*\mathbf{\mu}_b\mathbf{\mu}_a^T)=c\lambda_a^2
$$
>
>$$
\mathbf{q}\mathbf{k}_b^T=c(\lambda_a\lambda_b*\mathbf{\mu}_a\mathbf{\mu}_b^T + \lambda_b^2*\mathbf{\mu}_b\mathbf{\mu}_b^T)=c\lambda_b^2
$$
>
>When $\lambda_a \approx \lambda_b$,
>
>$$
\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)
$$
>
>When $\lambda_a \gg \lambda_b$ or $\lambda_a \ll \lambda_b$,
>
>$$
\textbf{a} \approx \mathbf{v}_a \text{ or } \textbf{a} \approx \mathbf{v}_b
$$
>
>Since randomly sampled following $\lambda_i \sim \mathcal{N}(1, \beta)$,
>
>$$
\mathbb{E}[\mathbf{q}\mathbf{k}_a^T]=\mathbb{E}[c\lambda_a^2]=c
$$
>
>$$
\mathbb{E}[\mathbf{q}\mathbf{k}_b^T]=\mathbb{E}[c\lambda_b^2]=c
$$
>
>Over multiple resamplings of $\lambda_1, ..., \lambda_m$,
>
>$$
\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)
$$

#### **1.4 Noisy Average with Multi-head Attention**
Q. Let's now consider a simple version of multi-head attention that averages the attended features resulting from two different queries. Here, two queries are defined ($\mathbf{q}_1$ and $\mathbf{q}_2$) leading to two different attended features ($\textbf{a}_1$ and $\textbf{a}_2$). The output of this computation will be $\textbf{a} = \frac{1}{2}(\textbf{a}_1 + \textbf{a}_2)$. Assume we have keys like those in Task 1.3, design queries $\mathbf{q}_1$ and $\mathbf{q}_2$ such that $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$.

>From the task 1.3, the expression $\mathbf{q}$ below,
>
>$$
\mathbf{q}=c(\mathbf{\mu}_a*\lambda_a + \mathbf{\mu}_b*\lambda_b)
$$
>
>This expression for $\mathbf{q}$ yields $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$. Utilizing the notion that each $\mathbf{\mu}_i$ can maintain its term, the following expressions for $\mathbf{q}_1$ and $\mathbf{q}_2$ can ensure $\textbf{a} = \frac{1}{2}(\textbf{a}_1 + \textbf{a}_2)$.
>
>$$
\mathbf{q}_1=c_1\lambda_a*\mathbf{\mu}_a
$$
>
>$$
\mathbf{q}_2=c_2\lambda_b*\mathbf{\mu}_b
$$
>
>By constructing the linear equation of $\mathbf{q}$ and $\mathbf{k}$,
>
>$$
\mathbf{q}_1\mathbf{k}_a^T=c_1\lambda_a*\mathbf{\mu}_a\mathbf{k}_a^T =c_1\lambda_a^2*(\mathbf{\mu}_a\mathbf{\mu}_a^T)=c_1\lambda_a^2
$$
>
>$$
\mathbf{q}_2\mathbf{k}_b^T=c_2\lambda_b*\mathbf{\mu}_b\mathbf{k}_b^T =c_2\lambda_b^2*(\mathbf{\mu}_b\mathbf{\mu}_b^T)=c_2\lambda_b^2
$$
>
>From here, $\textbf{a}_1$ and $\textbf{a}_2$ can be expressed,
>
>$$
\alpha_i=\mbox{softmax}(\mathbf{q}_1\mathbf{k}_i^T)
\hspace{20pt}
\alpha_j=\mbox{softmax}(\mathbf{q}_2\mathbf{k}_j^T)
$$
>
>$$
\alpha_i = 
\begin{cases} 
1 & \text{if } i = a \\
0 & \text{if } i \neq a 
\end{cases}
\hspace{20pt}
\alpha_j = 
\begin{cases} 
1 & \text{if } j = b \\
0 & \text{if } j \neq b
\end{cases}
$$
>
>$$
\textbf{a}_1 = \alpha_a \mathbf{v}_a = \mathbf{v}_a
\hspace{40pt}
\textbf{a}_2 = \alpha_b \mathbf{v}_b = \mathbf{v}_b
$$
>
>The final output is the average of $\textbf{a}_1$ and $\textbf{a}_2$.
>
>$$
\textbf{a} = \frac{1}{2}(\textbf{a}_1 + \textbf{a}_2)
$$

### 2. German-to-English Machine Translation

### 3. BLEU Score Comparision
