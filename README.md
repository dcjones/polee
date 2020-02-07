
# Polee

Polee is an alternative methodology for the analysis of RNA-Seq data. A lot
of RNA-Seq analysis does not consider the full likelihood, but instead takes
a two-step approach of first estimating transcript or gene expression then
treating those estimates as observations in subsequent analysis. This is more
efficient than a full probabilistic model encompassing all the reads in the
experiment, but does not fully account for uncertainty. To overcome this
shortcoming we make a compact and efficient approximation of the likelihood
function, and substitute it in place of the real thing. The end result is
more accurate analysis of RNA-Seq (e.g. differential expression) while
remaining tractable on inexpensive computers.

There will actual documentation in the near future ¯\\_(ツ)_/¯
