### Alloha The Application


## YGG's Refactoring Guide

- I've been developing in \_\_init\_\_.py files. 
    - Code should eventually be moved to modules.
- Boost class needs to be refactored. It's too big.
- Research modules should be removed
- Tests should be added for each module


## History

September 2023,
YGG Continues app development. The project begins to take shape. Refactoring is required.

August 2023,
YGG begins assembling a web application oriented towards exploring donations datasets, and tunable QF.

July 2023,
YGG creates a series of research notebooks that explore the donations dataset, the qf algorithm, the sme signal boosting, and advanced boosting with normalization and sigmoid applied.

June 2023,
Rxx creates a main.ipynb jupyter notebook that applies the tegr1 boost factor as the [following](https://discord.com/channels/810180621930070088/1050117836498018365/1136395276433760276):

```python
coefficient = 1 + 0.5 * (int(tec_tokens_flag) or int(tea_flag))
```

