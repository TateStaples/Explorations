# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Assignment 1 Solutions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 1 (ECCMMA 16.4)

    Consider the functions
    $$f(x_1, \dots, x_9) = f_1(x_1, x_3) f_2(x_1, x_4, x_5) f_3(x_1, x_6, x_7) f_4(x_4, x_8, x_9)$$

    and
    $$g_1(x_1) = \sum_{x_{\setminus 1}} f(x_1, \dots, x_9)$$

    Assume that each $x_i$ comes from an alphabet $\mathcal{X}$ with $|\mathcal{X}| = q$.

    * **(a)** Determine the number of computations required to compute $g_1(x_1)$ for all $x_1$ if the factorization is not used.
    * **(b)** Draw the factor graph associated with the given factorization of $f$.
    * **(c)** What is the complexity of computing $g_1(x_1)$ for all $x_1 \in \mathcal{X}$ with message passing if only the relevant messages are computed? Determine the messages passed on each edge of the tree for this computation.
    * **(d)** If only the relevant messages are computed, how many additional computations are required to compute:
    $$g_7(x_7) = \sum_{x_{\setminus 7}} f(x_1, \dots, x_9)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1. (ECCMMA 16.4) Complexity and Factor Graphs

    #### (a) Computations without factorization
    To compute $g_1(x_1)$ for all $x_1$, we must sum over $x_2, \dots, x_9$.
    Total variables $N=9$ and domain size $q$.

    Note that $x_2$ does not appear in the function $f$, which generally depends on $\{x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9\}$. Thus, $x_2$ is an isolated variable. Summing over $x_2$ simply contributes a scalar factor of $q$.

    The remaining summation involves 7 variables ($x_3, x_4, x_5, x_6, x_7, x_8, x_9$).
    Performing this naive summation for each of the $q$ values of $x_1$ requires iterating over $q^7$ configurations.

    **Complexity:** Approximately $O(q^9)$ if strictly iterating all variables, or $O(q^8)$ if the isolated variable is handled separately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (b) Factor Graph
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Here is the factor graph representation for the given factorization:
    $$f(x_1, \dots, x_9) = f_1(x_1, x_3) f_2(x_1, x_4, x_5) f_3(x_1, x_6, x_7) f_4(x_4, x_8, x_9)$$
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph TD
            x1((x1))
            x3((x3))
            x4((x4))
            x5((x5))
            x6((x6))
            x7((x7))
            x8((x8))
            x9((x9))

            f1[f1]
            f2[f2]
            f3[f3]
            f4[f4]

            x1 --- f1
            x3 --- f1

            x1 --- f2
            x4 --- f2
            x5 --- f2 

            x1 --- f3
            x6 --- f3
            x7 --- f3

            x4 --- f4
            x8 --- f4
            x9 --- f4
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    * **Variable Nodes:** $x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9$. (Factor graph usually shows variables in the scope).
    * **Factor Nodes:**
        * $f_1$ connected to $\{1, 3\}$
        * $f_2$ connected to $\{1, 4, 5\}$
        * $f_3$ connected to $\{1, 6, 7\}$
        * $f_4$ connected to $\{4, 8, 9\}$

    Structure is a tree. $x_1$ is a hub connected to $f_1, f_2, f_3$.
    $f_2$ connects to $x_4$, which connects to $f_4$.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph TD
            x1((x1)) --- f1[f1] --- x3((x3))
            x1 --- f2[f2]
            f2 --- x4((x4))
            f2 --- x5((x5))
            x4 --- f4[f4]
            f4 --- x8((x8))
            f4 --- x9((x9))

            x1 --- f3[f3]
            f3 --- x6((x6))
            f3 --- x7((x7))
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (c) Message Passing Complexity for $g_1(x_1)$
    We root at $x_1$ and pass messages in.
    Leaves: $x_3, x_5, x_6, x_7, x_8, x_9$.
    Messages:
    * $\mu_{3 \to 1}(x_1) = \sum_{x_3} f_1(x_1, x_3)$ ($q^2$)
    * $\mu_{f_3 \to 1}$ branch:
        - $6 \to f_3$ ($1$), $7 \to f_3$ ($1$)
        - $\mu_{f_3 \to 1}(x_1) = \sum_{x_6, x_7} f_3(x_1, x_6, x_7)$ ($q^3$)
    * $\mu_{f_2 \to 1}$ branch:
        - Need message from $x_4$.
        - $x_4$ needs message from $f_4$.
        - $8 \to f_4$, $9 \to f_4$.
        - $\mu_{f_4 \to 4}(x_4) = \sum_{x_8, x_9} f_4(x_4, x_8, x_9)$ ($q^3$)
        - $\mu_{4 \to f_2}(x_4) = \mu_{f_4 \to 4}(x_4)$
        - $5 \to f_2$.
        - $\mu_{f_2 \to 1}(x_1) = \sum_{x_4, x_5} f_2(x_1, x_4, x_5) \mu_{4 \to f_2}(x_4)$ ($q^3$)

    Final at $x_1$: Multiply incoming.
    Bottleneck complexity: Summing factors with 3 variables ($f_2, f_3, f_4$).
    Complexity: **$O(q^3)$**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (d) Additional for $g_7(x_7)$
    We need marginal at $x_7$.
    We have messages towards root $x_1$. Need messages towards $x_7$.
    Path: $x_1 \to f_3 \to x_7$.
    * Need $\mu_{1 \to f_3}$.
        - $x_1$ accumulates $\mu_{f_1 \to 1}$ and $\mu_{f_2 \to 1}$.
        - $\mu_{1 \to f_3}(x_1) = \mu_{f_1 \to 1}(x_1) \mu_{f_2 \to 1}(x_1)$ ($q$)
    * Need $\mu_{f_3 \to 7}(x_7) = \sum_{x_1, x_6} f_3(x_1, x_6, x_7) \mu_{1 \to f_3}(x_1)$ ($q^3$)

    Complexity: **$O(q^3)$**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 2 (ECCMMA 16.7)

    Let $p(x_1, x_2, x_3, x_4)$ be a joint probability mass function for the random variables $X_1, X_2, X_3, X_4$.

    * **(a)** Draw the (un) factor graph representing $p(x_1, x_2, x_3, x_4)$.
    * **(b)** Draw the factor graph representing standard decomposition:
    $$p(x_1, x_2, x_3, x_4) = p(x_1)p(x_2|x_1)p(x_3|x_1, x_2)p(x_4|x_1, x_2, x_3)$$

    * **(c)** Assume $X_1 \to X_2 \to X_3 \to X_4$ form a Markov chain and draw the factor graph associated with the factorization:
    $$p(x_1, x_2, x_3, x_4) = p(x_1)p(x_2|x_1)p(x_3|x_2)p(x_4|x_3)$$

    * **(d)** Suppose that the $X_i$ are observed indirectly through $Y_i$. Draw the factor graph for the hidden Markov model (HMM) implied by the factorization:
    $$p(y_{1:N}, x_{1:N}) = p(y_1) \prod_{i=2}^N p(y_i|y_{i-1}) \prod_{i=1}^N p(x_i|y_i)$$
    where it is assumed that $p(x_i|y_{1:N}, x_{\setminus i}) = p(x_i|y_i)$ (e.g., $X_i$ is independent of $Y_{\setminus i}$ given $Y_i$).

    ---

    ### 2. (ECCMMA 16.7) Factor Graphs and HMMs
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (a) (Un)factor graph
    Representing $p(x_1, x_2, x_3, x_4)$ without specific structure requires a single factor node connected to all four variable nodes $x_1, x_2, x_3, x_4$.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph TD
            x1((x1)); x2((x2)); x3((x3)); x4((x4))
            F[P]
            F --- x1; F --- x2; F --- x3; F --- x4
            style F fill:#eee,stroke:#333,shape:rect
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (b) Standard Decomposition
    Factorization: $p(x_1, x_2, x_3, x_4) = p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) p(x_4|x_1, x_2, x_3, x_4)$.

    Graph has 4 factor nodes creating a fully connected structure.
    """)
    return


@app.cell
def _(mo):

    mo.mermaid(
        r"""
        graph TD
            x1((x1)); x2((x2)); x3((x3)); x4((x4))
            f1[f1]; f2[f2]; f3[f3]; f4[f4]
            f1 --- x1
            f2 --- x1; f2 --- x2
            f3 --- x1; f3 --- x2; f3 --- x3
            f4 --- x1; f4 --- x2; f4 --- x3; f4 --- x4
            style f1 fill:#eee,stroke:#333,shape:rect
            style f2 fill:#eee,stroke:#333,shape:rect
            style f3 fill:#eee,stroke:#333,shape:rect
            style f4 fill:#eee,stroke:#333,shape:rect
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (c) Markov Chain
    Factorization: $p(x_{1:4}) = p(x_1) p(x_2|x_1) p(x_3|x_2) p(x_4|x_3)$.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph LR
            x1((x1)) --- f2[f2] --- x2((x2)) --- f3[f3] --- x3((x3)) --- f4[f4] --- x4((x4))
            f1[f1] --- x1
            style f1 fill:#eee,stroke:#333,shape:rect
            style f2 fill:#eee,stroke:#333,shape:rect
            style f3 fill:#eee,stroke:#333,shape:rect
            style f4 fill:#eee,stroke:#333,shape:rect
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (d) Hidden Markov Model (HMM)
    Factorization: $p(y_{1:N}, x_{1:N}) = p(y_1) \prod_{i=2}^N p(y_i|y_{i-1}) \prod_{i=1}^N p(x_i|y_i)$.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph TD
            y1((y1)); y2((y2)); y3((y3)); y4((y4))
            x1((x1)); x2((x2)); x3((x3)); x4((x4))

            f_init["p(y1)"] --- y1

            y1 --- f_t1[trans] --- y2
            y2 --- f_t2[trans] --- y3
            y3 --- f_t3[trans] --- y4

            y1 --- f_e1[emit] --- x1
            y2 --- f_e2[emit] --- x2
            y3 --- f_e3[emit] --- x3
            y4 --- f_e4[emit] --- x4

        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 3 (FGML 1.3)

    A TA wants to graduate (G), which depends on papers (P) and proposal (R). Papers and theorems (T) are generated by drinking coffee (C). Coffee depends on the university providing it (U) and their favorite coffee shop, Tazza D'Oro (D), being open. All variables are binary {T, F}.

    **CPT Parameters:**

    *   $P(U=T) = 0.1$
    *   $P(D=T|U=T) = 0.6, \quad P(D=T|U=F) = 0.5$
    *   $P(C=T|U=T, D=T) = 0.7, \quad P(C=T|U=T, D=F) = 0.5$
    *   $P(C=T|U=F, D=T) = 0.6, \quad P(C=T|U=F, D=F) = 0.05$
    *   $P(P=T|C=T) = 0.8, \quad P(P=T|C=F) = 0.6$
    *   $P(T=T|C=T) = 0.7, \quad P(T=T|C=F) = 0.6$
    *   $P(R=T|P=T) = 0.7, \quad P(R=T|P=F) = 0.1$
    *   $P(G=T|P=T,R=T) = 0.9, \quad P(G=T|P=T,R=F) = 0.3$
    *   $P(G=T|P=F,R=T) = 0.5, \quad P(G=T|P=F,R=F) = 0.1$

    **Graduation Probabilities (Explicit):**
    * $P(G=T|P=T, R=T) = 0.9$
    * $P(G=T|P=T, R=F) = 0.3$
    * $P(G=T|P=F, R=T) = 0.5$
    * $P(G=T|P=F, R=F) = 0.1$

    **Inference Tasks:**
    * **(a)** How likely is the TA to graduate if her coffee supply runs out? $P(G = T | C = F) = ?$
    * **(b)** How likely is the TA to graduate if Tazzo is open all the time? $P(G = T | D = T) = ?$
    * **(c)** Should we even be worried about her coffee supply running out? $P(C = T) = ?$
    * **(d)** If she is writing theorems, does this mean she will also publish? $P(P = T | T = T) = ?$
    * Report the ordering and factors produced after eliminating each variable for query (a).
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph TD
            U((Univ)) --> D(("Tazza D'Oro"))
            U --> C((Coffee))
            D --> C
            C --> T((Theorems))
            C --> P((Papers))
            P --> R((Propose))
            P --> G((Graduate))
            R --> G

            style U fill:#fff,stroke:#333
            style D fill:#fff,stroke:#333
            style C fill:#fff,stroke:#333
            style T fill:#fff,stroke:#333
            style P fill:#fff,stroke:#333
            style R fill:#fff,stroke:#333
            style G fill:#fff,stroke:#333
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### 3. (FGML 1.3) Solution
    """)
    return


@app.cell
def _(mo):
    # Parameters
    p_U_T = 0.1
    p_U_F = 1 - p_U_T

    # P(D=T|U)
    p_D_T_given_U_T = 0.6
    p_D_T_given_U_F = 0.5

    # P(C=T | U, D)
    p_C_T_UT_DT = 0.7
    p_C_T_UT_DF = 0.5
    p_C_T_UF_DT = 0.6
    p_C_T_UF_DF = 0.05

    # P(P=T|C)
    p_P_T_given_C_T = 0.8
    p_P_T_given_C_F = 0.6

    # P(T=T|C)
    p_T_T_given_C_T = 0.7
    p_T_T_given_C_F = 0.6

    # P(R=T|P)
    p_R_T_given_P_T = 0.7
    p_R_T_given_P_F = 0.1

    # Helper to iterate all states
    states = [0, 1] # F=0, T=1

    prob_C_T = 0
    prob_G_on_C_F = 0
    prob_C_F = 0
    prob_G_on_D_T = 0
    prob_D_T = 0
    prob_P_T_T_T = 0
    prob_T_T = 0

    for u in states:
        p_u = p_U_T if u else p_U_F
        for d in states:
            if u:
                p_d = p_D_T_given_U_T if d else (1 - p_D_T_given_U_T)
            else:
                p_d = p_D_T_given_U_F if d else (1 - p_D_T_given_U_F)

            for c in states:
                if u and d: p_c = p_C_T_UT_DT if c else (1 - p_C_T_UT_DT)
                elif u and not d: p_c = p_C_T_UT_DF if c else (1 - p_C_T_UT_DF)
                elif not u and d: p_c = p_C_T_UF_DT if c else (1 - p_C_T_UF_DT)
                else: p_c = p_C_T_UF_DF if c else (1 - p_C_T_UF_DF)

                for p in states:
                    p_p = p_P_T_given_C_T if c else p_P_T_given_C_F
                    if not p: p_p = 1 - p_p

                    for t_node in states:
                        p_t = p_T_T_given_C_T if c else p_T_T_given_C_F
                        if not t_node: p_t = 1 - p_t

                        for r in states:
                            p_r = p_R_T_given_P_T if p else p_R_T_given_P_F
                            if not r: p_r = 1 - p_r

                            for g in states:
                                if p and r: p_g = 0.9
                                elif p and not r: p_g = 0.3
                                elif not p and r: p_g = 0.5
                                else: p_g = 0.1
                                if not g: p_g = 1 - p_g

                                joint = p_u * p_d * p_c * p_p * p_t * p_r * p_g

                                # (c) P(C=T)
                                if c: prob_C_T += joint
                                else: prob_C_F += joint

                                # (a) P(G=T | C=F)
                                if c == 0:
                                    if g == 1:
                                        prob_G_on_C_F += joint

                                # (b) P(G=T | D=T)
                                if d == 1:
                                    prob_D_T += joint
                                    if g == 1:
                                        prob_G_on_D_T += joint

                                # (d) P(P=T | T=T)
                                if t_node == 1:
                                    prob_T_T += joint
                                    if p == 1:
                                        prob_P_T_T_T += joint

    mo.md(fr"""
    #### Calculated Results:

    * **(a) P(G=T \| C=F)** = {prob_G_on_C_F / prob_C_F:.4f}
    * **(b) P(G=T \| D=T)** = {prob_G_on_D_T / prob_D_T:.4f}
    * **(c) P(C=T)** = {prob_C_T:.4f}
    * **(d) P(P=T \| T=T)** = {prob_P_T_T_T / prob_T_T:.4f}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Efficiency Order:
    When eliminating for $P(G|C=F)$, we can use order: $U, D, T, P, R$.
    1. **Eliminate $U$**: Sum $\sum_U P(U)P(D|U)$ to produce factor $\tau_1(D)$.
    2. **Eliminate $D$**: Sum $\sum_D \tau_1(D) P(C=F|U,D)$. Note that the factor $P(C|U,D)$ links $U, D, C$. After eliminating $U$, we have a factor on $D, C$. Since $C$ is observed ($C=F$), this becomes a factor on $D$ alone. Eliminating $D$ yields a scalar.
    3. **Eliminate $T$**: The factor $P(T|C)$ depends only on $C$. Since $C$ is observed, $\sum_T P(T|C=F) = 1$, so this branch sums to 1.
    4. **Eliminate $P$**: We compute $\tau_P(R, G) = \sum_P P(P|C=F)P(R|P)P(G|P,R)$. This involves summing a product of three terms over $P$, producing a factor on $\{R, G\}$.
    5. **Eliminate $R$**: Finally, compute $\tau_R(G) = \sum_R \tau_P(R, G)$.
    6. **Normalize**: Normalize the resulting factor $\tau_R(G)$ to get $P(G|C=F)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 4 (MCT 2.4)

    Schoolchildren form a tree graph where they only see/communicate with nearest neighbors.

    * Construct a message-passing algorithm to count all classmates in a distributed fashion.
    * What is the initialization and what are the message-passing rules?
    * How can you modify it to only count a subset (e.g., girls)?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4. (MCT 2.4) Distributed Counting Algorithm
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Initialization & Rules
    To count the total number of nodes $N$ in a tree using message passing:

    1. **Initialization:** Every child (leaf node) initializes a message with value 1 (counting itself).


    2. **Message Passing Rule ($i \to j$):**
    Node $i$ sends to neighbor $j$:
    $$\mu_{i \to j} = 1 + \sum_{k \in N(i) \setminus j} \mu_{k \to i}$$


    > (My count = 1 + sum of counts from all my other neighbors).

    3. **Determination:** Any node can compute the total count by summing all incoming messages plus 1: $N = 1 + \sum_{k \in N(i)} \mu_{k \to i}$.
    4. **Wolf Detection:** If the "group" forms a tree, checking the total count against the expected class roster (e.g., $N_{expected}$) reveals if anyone is missing. If the graph is disconnected (eaten children broke links), the count will be $< N_{expected}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Modification for Subset (Girls)
    Modify the local contribution term.

    * Let $c_i = 1$ if child $i$ is a girl, else 0.
    * Message: $\mu_{i \to j} = c_i + \sum_{k \in N(i) \setminus j} \mu_{k \to i}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 5 (FGML 1.5)

    Model independent sets $x_1, \dots, x_N$ on an undirected graph $G=(V,E)$.

    * **(a)** Use binary variables $x_i \in \{0, 1\}$ ($x_i=1$ if chosen). Define pairwise factors $\psi_{ij}(x_i, x_j)$ such that $\prod_{(i,j)\in E} \psi_{ij}$ is 1 if $x$ is an independent set and 0 otherwise.
    * **(b)** Explain why the partition function $Z = \sum_x \prod \psi_{ij}$ counts the number of independent sets.
    * **(c) Weighted Variant:** With $w > 0$, define $p(x) \propto w^{\sum_i x_i} \prod_{(i,j)} \psi_{ij}$.
        * i. Interpret $Z(w)$ as a generating function.
        * ii. Express $E[Size]$ in terms of $Z(w)$.
    * **(d) Computation on Tree (Fig 1):** Root the tree at vertex 1.
        * i. Write $\mu_{4 \to 2}$ and $\mu_{5 \to 2}$ explicitly.
        * ii. Compute $\mu_{2 \to 1}$ as a simplified polynomial in $w$.
        * iii. Evaluate at $w=1$.
        * iv. Compute the marginal $p(x_1=1)$.
        * v. Describe how to efficiently sample independent sets.

    ---

    ### 5. (FGML 1.5) Independent Sets
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (a) Pairwise Factor Graph
    Let $x_i \in \{0, 1\}$. Define pairwise factors for edges $(i, j) \in E$:
    $$\psi_{ij}(x_i, x_j) = \begin{cases} 0 & \text{if } x_i=1, x_j=1 \\ 1 & \text{otherwise} \end{cases} = \text{NAND}(x_i, x_j)$$

    The function $f(x) = \prod_{(i,j)} \psi_{ij}(x_i, x_j)$ is 1 if and only if no two adjacent vertices are both 1 (valid independent set), and 0 otherwise.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (b) Partition Function
    $$Z = \sum_x \prod_{(i,j)} \psi_{ij}(x_i, x_j)$$
    The product of $\psi_{ij}$ over all edges $(i,j)$ is 1 if and only if $x$ is an independent set, and 0 otherwise. Summing over all possible assignments $x$ counts the number of independent sets.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (c) Weighted Variant (Generating Function)
    With $\phi_i(x_i) = \lambda^{x_i}$:
    $$p_\lambda(x) = \frac{1}{Z(\lambda)} \prod_i \lambda^{x_i} \prod_{(i,j)} \psi_{ij}(x_i, x_j) = \frac{1}{Z(\lambda)} \lambda^{\sum x_i} \mathbb{I}(x \in IS)$$

    **i. Generating Function:**
    The partition function sums the unnormalized probability weight over all possible configurations $x$:
    $$Z(\lambda) = \sum_x \lambda^{\sum x_i} \mathbb{I}(x \in IS)$$
    Group the terms by the size of the independent set $k = \sum x_i$:
    $$Z(\lambda) = \sum_{k=0}^{|V|} N_k \lambda^k$$
    where $N_k$ is the number of independent sets of size $k$. Thus, $Z(\lambda)$ is the **generating function** for the counts of independent sets, with $\lambda$ acting as the formal variable.

    **ii. Expectation of Size:**
    Let $|S| = \sum_i X_i$ be the size of the random independent set.
    $$E[|S|] = \sum_x \left(\sum_i x_i\right) p_\lambda(x) = \frac{1}{Z(\lambda)} \sum_x \left(\sum_i x_i\right) \lambda^{\sum x_i} \mathbb{I}(x \in IS)$$
    Notice that $\frac{\partial}{\partial \lambda} (\lambda^k) = k \lambda^{k-1}$, so $\lambda \frac{\partial}{\partial \lambda} (\lambda^k) = k \lambda^k$.
    Applying this operator to $Z(\lambda)$:
    $$\lambda \frac{\partial}{\partial \lambda} Z(\lambda) = \sum_x \left(\sum_i x_i\right) \lambda^{\sum x_i} \mathbb{I}(x \in IS)$$
    Therefore:
    $$E[|S|] = \frac{1}{Z(\lambda)} \left( \lambda \frac{\partial Z(\lambda)}{\partial \lambda} \right) = \lambda \frac{\partial}{\partial \lambda} \log Z(\lambda)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (d) Exact Computation on Figure 1
    Tree: Root 1, children 2,3. Children of 2 are 4,5. Children of 3 are 6,7.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        r"""
        graph TD
            1((1)) --- 2((2))
            1 --- 3((3))
            2 --- 4((4))
            2 --- 5((5))
            3 --- 6((6))
            3 --- 7((7))
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (e) Messages $4 \to 2$
    $$m_{4 \to 2}(x_2) = \sum_{x_4} \lambda^{x_4} \psi(x_4, x_2)$$

    * If $x_2=0$: $x_4$ can be 0 (value 1) or 1 (value $\lambda$), so the sum is $1 + \lambda$.
    * If $x_2=1$: $x_4$ must be 0 (value 1), so the sum is $1$.

    Message vector: $[1+\lambda, 1]$. By symmetry, the message $5 \to 2$ is identical.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (f) $Z(\lambda)$
    Message $2 \to 1$:
    $$m_{2 \to 1}(x_1) = \sum_{x_2} \lambda^{x_2} \psi(x_2, x_1) m_{4 \to 2}(x_2) m_{5 \to 2}(x_2)$$

    * For $x_1=0$:
        * $x_2=0$: Term is $1 \cdot 1 \cdot (1+\lambda)^2 = (1+\lambda)^2$.
        * $x_2=1$: Term is $\lambda \cdot 1 \cdot (1)^2 = \lambda$.
        * Total: $m_{2 \to 1}(0) = (1+\lambda)^2 + \lambda = 1 + 3\lambda + \lambda^2$.

    * For $x_1=1$:
        * $x_2=0$: Term is $1 \cdot 1 \cdot (1+\lambda)^2 = (1+\lambda)^2$.
        * $x_2=1$: Constraint violation ($\psi=0$), term is 0.
        * Total: $m_{2 \to 1}(1) = (1+\lambda)^2$.

    At Root 1:
    $$Z(\lambda) = \sum_{x_1} \lambda^{x_1} m_{2 \to 1}(x_1) m_{3 \to 1}(x_1)$$
    (Node 3 branch is identical to 2).
    $x_1=0$: $1 \cdot ((1+\lambda)^2+\lambda)^2$.
    $x_1=1$: $\lambda \cdot ( (1+\lambda)^2 )^2$.

    $$Z(\lambda) = (1 + 3\lambda + \lambda^2)^2 + \lambda(1 + 2\lambda + \lambda^2)^2$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (g) Evaluate at $\lambda=1$
    $Z(1) = (5)^2 + 1(4)^2 = 25 + 16 = 41$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (h) Marginal $p(X_1=1)$
    $$p(X_1=1) = \frac{\lambda (1+\lambda)^4}{Z(\lambda)}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### (i) Sampling
    Standard root-to-leaf sampling.
    1. Sample $x_1 \sim p(x_1)$.
    2. Given $x_i$, sample neighbor $x_j$ with probs proportional to $\lambda^{x_j} \psi(x_i, x_j) \times \text{incoming msgs from below}$.
    """)
    return


if __name__ == "__main__":
    app.run()
