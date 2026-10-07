Public AIMO 2025 reference problem `a1d40b` from `aimo-interp/val-sample`.

**Original**

The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$. There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. How many prime factors does $N$ have, counted with multiplicity?

**BAS v1 `typos`**

The Fiboncaci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$. There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. How many prime factors does $N$ have, counted with multpilicity?

**BAS v1 `distract`**

The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$. There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. How many prime factors does $N$ have, counted with multiplicity?

(Unrelated note: this problem was typed up by a volunteer on a quiet afternoon and has been checked by two readers.)

**BAS v1 `rephrase`**

Here is a competition problem.

The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$. There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. Determine how many prime factors does $N$ have, counted with multiplicity?

Solve it carefully.

**BAS v1 `rename`** (unused in v1; falls back to rephrase(typos) when no name is present)

Here is a competition problem.

The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$. There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. Determine how many prime factors does $N$ have, counted with multpilicity?

Solve it carefully.

**Organisers' `distract` variant for the same problem** (from the `permutations_causing_decay` field of the `gpt-oss-120b:low` row)

The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$ (as a minor note, the recursion may be checked mod 5 using the repeating Pisano period). There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. How many prime factors does $N$ have, counted with multiplicity?
