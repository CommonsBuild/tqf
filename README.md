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
