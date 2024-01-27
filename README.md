# Tunable Quadratic Funding
TQF is a research initiative by the Token Engineering Commons that enables
transparent and informed quadratic funding round operation with a focus on
utilizing token signal inputs to add tunable weights to donations based on
donor token holdings.


## Table of Contents
1. [Background](#background)
2. [Purpose](#purpose)
3. [Scope of the Project](#scope-of-the-project)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [Testing and Feedback](#testing-and-feedback)
8. [Specification](#specification)
9. [TEGR](#tegr)
10. [Acknowledgements](#acknowledgements)
11. [Contact Information](#contact-information)

## Background

The Token Engineering Commons has been running community Quadratic Funding on
the Grants Stack since the Beta round of April 2023. Since the inception of the
Token Engineering Grants Rounds
([TEGR](https://forum.tecommons.org/c/grants-program/32)), an experimental
feature was introduced called subject matter expertise signal boosting
([SMESB](https://medium.com/token-engineering-commons/expertise-and-quadratic-funding-bd4f0c5c3e23)).
In SMESB donations weights are boosted according to a SME boost weight assigned
to donor addresses. In TEGR, donor weights are computed as a combination of TE
academy credentials and $TEC token holdings. These signals are meant to
indicate expertise in the field of Token Engineering.

For more information on the background of TQF, see the original blog post:  
[Incorporating Expertise into Quadratic Funding.](https://medium.com/token-engineering-commons/expertise-and-quadratic-funding-bd4f0c5c3e23)


## Purpose

The purpose of this research repository is to provide a data science research
environment to the operators of the TEGR rounds and a tool for other community
round operators that wish to employ the techniques being highlighted as tunable
quadratic funding.

Prospective Benefits of Tunable Quadratic Funding Include:
* Increased Sybil Resistance
* Subject Matter Expertise Signal Boosting
* Community Token Signal Processing
* Pluralistic Framework for Analysing Funding Public Goods
* Platform for Quadratic Funding Research Education and Communication

The TQF tool allows for stepping through the process of quadratic funding and
understanding the deeper implications of applying token signals as boosts to
donations. This process allows communities to explore the alignment between
resource allocation and values. As a general contribution to public goods
tooling, we expect this tool to aid in attracting funding to the domain of
token engineering public goods and the greater public goods funding ecosystem.

## Scope of the Project

Proposed Delivery for Q1 2024:  
* A paper or extensive forum post with our findings, recommendations and a
framework to tune QF at the end of this proposal’s period.
* An open-sourced MVP tool for all operators to be able to tune QF.
* Training materials that allow grant operators to confidently apply tunable QF
to their community grants rounds.


## Getting Started

TQF is implemented in Python using the [HoloViz](https://holoviz.org) data
science stack. Dependency management is handled by [Python
Poetry](https://python-poetry.org/).

Installation Requirements
* Python3.10
* Python Poetry
* Git

To run this app locally follow the steps below:
1. Clone the [repository](https://github.com/CommonsBuild/tqf) and checkout the development branch.
```
git clone git@github.com:CommonsBuild/tqf.git
cd tqf
git checkout ygg
```
2. Install the dependencies and activate python environment
```
poetry install
poetry shell
```
3. Run the app
```
panel serve tqf/app.py
```

The app should now be running. Navigite to http://localhost:5006/app in your browser.

## Usage
The framework takes in donation datasets, token distributions, and user
input to compute the final funding allocation for each project.

The operations required to utilize TQF are the following:
1. Input a donation dataset
2. Input token distribution datasets
3. Configure the parameters of your boosts

The above steps can be done either programmatically or in the GUI

## Contributing 

The project is built using the [HoloViz](https://holoviz.org) data science
stack with primary heavy lifting from, [Panel](https://panel.holoviz.org/),
[hvplot](https://hvplot.holoviz.org/user_guide/Plotting.html), and
[Tabulator](https://panel.holoviz.org/reference/widgets/Tabulator.html). If you
are familiar with these tools or would like to learn, please consider taking a
look at contributing to the project. 


You can get started contributing by picking up
[issues](https://github.com/CommonsBuild/tqf/issues) on this repository.

## Testing and Feedback
It is very valuable for us to receive feedback on our work. Please
[open an issue](https://github.com/CommonsBuild/tqf/issues) if you have any
questions or topics of discussion that you would like to bring to our attention.
Please get in touch with



## Specification

### Quadratic Funding

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
radicalized element wise and then project columns are summed on axis 0. The
resulting vector is squared element wise to give the quadratic funding per
project. The funding outcome is then normalized such that it sums to 1 and represents a
distribution to be made by the matching pool.


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

<div align="center">
<center>

| Address                           | Balance |
|-----------------------------------|---------|
| 0x456...abc                       | 200     |
| 0x123...def                       | 100     |
| ...                               | ...     |

</center>
</div>

The dataset can represent fungible or non-fungible tokens.


### Token Signaling

Given a token distribution, a boost vector can be created using a
signal transformation.

 The donor coefficients are applied to the donations as part of the QF
 calculation, ensuring that each donor's influence is weighted according to the
 community-defined boost vector $B$.

The framework is flexible and can accommodate various methods of determining
the donor coefficients, such as:

- Historical contribution analysis
- Token holdings snapshots
- Community voting mechanisms
- Other custom algorithms designed by the community

By providing this level of customization, TQF empowers communities to experiment with and optimize their funding mechanisms, leading to more equitable and effective public goods funding.


## TEGR

### TEGR2

### TEGR1

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

#### History

September 2023,
YGG Continues app development. The project begins to take shape. Refactoring is required.

August 2023,
YGG begins assembling a web application oriented towards exploring donations datasets, and tunable QF.

July 2023,
YGG creates a series of research notebooks that explore the donations dataset, the qf algorithm, the sme signal boosting, and advanced boosting with normalization and sigmoid applied.

June 2023,
Rxx creates a main.ipynb jupyter notebook that applies the tegr1 boost factor as the [following](https://discord.com/channels/810180621930070088/1050117836498018365/1136395276433760276):

Original TQF Formula:

```python
coefficient = 1 + 0.5 * (int(tec_tokens_flag) or int(tea_flag))
```



## Acknowledgements

This research repository is maintained by [The Token Engineering Commons
(TEC)](https://twitter.com/tecmns/) to aid in the operation of the Token
Engineering Grant Round (TEGR) series which allocates a target annual
$100,000USD funding to token engineering public goods projects via Quadratic
Funding.

Funding is provided by TEC Coordination team, and YGG as per the TEC Data Science Fellowship.
* https://forum.tecommons.org/t/4-month-te-data-science-fellowship/1287
* https://forum.tecommons.org/t/tec-coordination-team-operating-budget-sep-dec-2023/1286

This research repository was initialized Rxx from the TEC Tech Team.


## Contact Information
Contact us on twitter.
* [@tecmns](https://twitter.com/tecmns)
* [@ygg_anderson](https://twitter.com/ygg_anderson)
* [@entigdd](https://twitter.com/entigdd)
* [@8ctopuso](https://twitter.com/8ctopuso)

Join the Weekly Open Development Call in TEC Discord Thursdays 12:00-1:00pm PST
