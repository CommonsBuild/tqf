# Alloha


This research repository is maintained by The Token Engineering Commons (TEC) to aid in the
operation of the Token Engineer grant round series (TEGRX) which allocates a target $100,000USD
annual funding to token engineering public goods projects via Quadratic Funding. This research is funded from the TEC Common Pool via TAO Voting as an
initiative lead by the TEC Coordination Team and YGG the data science fellow at
TEC.

This repository was initialized by TEC technical support team (Rxx) to apply
TEC token distribution and TEA credentials as a signal for TE expertise to by
applied in QF funding rounds. To learn more, read the following:    
https://medium.com/token-engineering-commons/expertise-and-quadratic-funding-bd4f0c5c3e23

This repository now contains Alloha, a data science web framework that serves
communities in operating the Tunable Quadratic Funding process.

## Tunable Quadratic Funding

Tunable QF is a process that combines donation and token distribution datasets
in a signal processing environment using the python data science stack. A token
distribution represents a snapshot of a particular fungible or non-fungible
token contract. A donation dataset contains rows with donor, project,
amountUSD, and timestamp. Token distributions can be used to assign
coefficients to donors or to projects. 

In the case of the TEGR1 algorithm, each donation is multiplied by the donor
coefficient for that donation such that the qf algorithm becomes:

## Modified Quadratic Funding Formula

Quadratic Funding (QF) is a mathematical framework used to allocate funds to public goods in a way that optimally balances the number of contributions with the size of contributions. In the modified version of the QF used by the TEGRX series, each donation amount is adjusted by a donor-specific coefficient, \( c_i \), which represents the weight or influence of that donor's contribution. This coefficient can be determined by various factors such as past contributions, reputation, or other metrics relevant to the community.

The modified QF formula with the donor coefficient is given by:

$$
F = \left( \sum_{i=1}^{n} \sqrt{c_i \cdot d_i} \right)^2
$$

Where:

- \( F \) is the total funding that a project receives after the QF round.
- \( c_i \) is the donor coefficient for the \( i \)-th donation.
- \( d_i \) is the amount of the \( i \)-th donation.
- \( n \) is the total number of donations.

The donor coefficients can be represented as a vector:

$$
\vec{c} = \begin{bmatrix}
           c_1 \\
           c_2 \\
           \vdots \\
           c_n
         \end{bmatrix}
$$

This vector \( \vec{c} \) allows us to adjust the influence of each donation individually, providing a tunable lever to the QF mechanism. By adjusting the coefficients, the community can fine-tune the funding distribution to better reflect its values and goals.

### Implementation in Alloha

In Alloha, the tunable QF process is implemented using the Python data science stack. The framework takes in donation datasets and token distributions to compute the final funding allocation for each project. The donor coefficients are applied to the donations as part of the QF calculation, ensuring that each donor's influence is weighted according to the community-defined vector \( \vec{c} \).

The framework is flexible and can accommodate various methods of determining the donor coefficients, such as:

- Historical contribution analysis
- Token holdings snapshots
- Community voting mechanisms
- Other custom algorithms designed by the community

By providing this level of customization, Alloha empowers communities to experiment with and optimize their funding mechanisms, leading to more equitable and effective public goods funding.

---

For more information on how to use Alloha and contribute to the research, please refer to the [Documentation](#) and [Contribution Guidelines](#).












# TEC Matching Processing

This repo encompasses the Jupyter notebooks we have used in order to determine matching exponents for all individuals.

We have used two Dune queries prepared before the fact and alpha round data to create the initial process, and have amended it now that the round data is available.

You will find all the necessary information, statistics and process in the ./main.ipynb file.

## Notes

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
