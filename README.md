# Alloha
---

For more information on how to use Alloha and contribute to the research,
please refer to the [Documentation](#) and [Contribution Guidelines](#).

This research repository is maintained by The Token Engineering Commons (TEC)
to aid in the operation of the Token Engineering Grant Round (TEGRX) series
which allocates a target annual $100,000USD funding to token engineering
public goods projects via Quadratic Funding. To learn more, read the following:    
https://medium.com/token-engineering-commons/expertise-and-quadratic-funding-bd4f0c5c3e23

## Quadratic Funding

Quadratic Funding is a capital allocation protocol that determines the
distribution of matching funds across a set of public goods projects. The 
algorithm determines the funding outcome based off of peer to peer contributions
that are made from citizens to public goods. Formally:

```math
\mathbf{F}_p = \left( \sum_{i=1}^{n} \sqrt{c_{ip}} \right)^2
```

Where $`\mathbf{F}_{p}`$ is the quadratic funding for project $`p`$ and $`c_{ip}`$ is the contribution made 
by citizen $`c_i`$ to project $`p`$ where there are $`n`$ citizens. In matrix notation, QF is an operation that maps a contributions matrix
$`\mathbf{C} \in \mathbb{R}^{n \times m}`$ to a funding vector $`\mathbf{F} \in
\mathbb{R}^{m}`$ given $`m`$ public goods projects. 

```math
\mathbf{F} = \left( \text{sum}\left( \sqrt{\mathbf{C}} \right) \right)^2
```

The contributions matrix is
radicalized elementwise and then project columns are summed on axis 0. The
resulting vector is squared elementwise to give the quadratic funding per
project. The funding outcome is then normalized such that it sums to 1 and represents a
distribution to be made by the matching pool.


## Tunable Quadratic Funding

### Contributor Boost Coefficient

Tunable QF introduces a contributor boost coefficient $`b_i`$ for each citizen
$`c_i`$  that is applied as a multiplicative factor to each donation $`c_{ip}`$ for
each public good $`p`$, such that the resulting funding mechanism becomes:



```math
\mathbf{F}_p = \left( \sum_{i=1}^{n} \sqrt{b_i\cdot{c_{ip}}} \right)^2
```


In matrix notation, we are applying the boost vector $`\mathbf{B} \in
\mathbb{R}^{n}`$ as a coefficient for the contribution matrix.


```math
\mathbf{F} = \left( \sqrt{\mathbf{B}^T}\cdot{\sqrt{\mathbf{C}}}  \right)^2
```

Notice above that we do not need the sum operator anymore due to the nature of vector matrix multiplication.


### Token Balances

Consider a token distribution dataset as a vector $`\mathbf{T} \in 
\mathbb{Z}^{n+}`$ such that $`\mathbf{T_i}`$ is the balance of $`c_i`$.

| Address                           | Balance |
|-----------------------------------|---------|
| 0x456...abc                       | 200     |
| 0x123...def                       | 100     |
| ...                               | ...     |



The dataset can represent fungible or non-fungible tokens.


### Token Signaling

Given a token distribution, a boost vector can be created using a
signal transformation.

### Implementation in Alloha

In Alloha, the tunable QF process is implemented using a Python data science
stack. The framework takes in donation datasets and token distributions to
compute the final funding allocation for each project. The donor coefficients
are applied to the donations as part of the QF calculation, ensuring that each
donor's influence is weighted according to the community-defined boost vector $B$.

The framework is flexible and can accommodate various methods of determining
the donor coefficients, such as:

- Historical contribution analysis
- Token holdings snapshots
- Community voting mechanisms
- Other custom algorithms designed by the community

By providing this level of customization, Alloha empowers communities to experiment with and optimize their funding mechanisms, leading to more equitable and effective public goods funding.



## TEGR1

This repo encompasses the Jupyter notebooks we have used in order to determine
matching exponents for all individuals. We have used two Dune queries prepared
before the fact and alpha round data to create the initial process, and have
amended it now that the round data is available. You will find all the
necessary information, statistics and process in the ./main.ipynb file.

### Notes

- update matching score if:
  - holds 10 TEC
    or
  - TE Academy Certificate
    (snapshot as of 2023-05-09 23:50 UTC)
    (based on a combination of):
    https://dune.com/queries/2457581
    plus extracts from TEA
- generate report on round statistics and effect on TE (percentage who have tec / te certificate and donate)
  - ever since the TE Round was announced and carried out, the $TEC price has stabilised and grown.
  - generate chart for unique holders
    https://dune.com/queries/2457553/4040451
