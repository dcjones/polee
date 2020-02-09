
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


# Installing

Polee is installable through the Julia package manager, but is meant to be run
primarily as a command line script, so there are a couple extra steps involved.

First install the julia package like so
```julia
using Pkg
Pkg.add(PackageSpec(name="Polee", url="https://github.com/dcjones/polee.git"))
```

Then install the command line script. By default this will put it in
`$HOME/bin`, but an alternative path can be passed to the `install_polee_script` function.
```julia
using Polee
install_polee_script()
```


# Using


## Approximating likelihood

Before running models, each sample must be "prepared", by approximating the
likelihood function and storing a representation.

Likelihood approximation assumes reads are alignments in BAM files, which
need not be sorted, and can be alignments to a transcriptome to genome. If
aligned to a genome you must also provide transcript annotations as GFF3
file.

If using genome alignments, the command looks like

```sh
polee prep-sample -o prepared-sample.h5 genome.fa reads.bam annotations.gff3
```

And if using transcriptome alignments,

```sh
polee prep-sample -o prepared-sample.h5 transcriptome.fa reads.bam
```

Either of these commands will output a file `prepared-sample.h5` which should
be just a few megabytes and be a self-contained and fairly accurate
approximation of the likelihood function. You won't need the reads or
alignments after this.

There are few optional arguments to this command, but most aren't of interest
in normal use. For example, `--no-bias` will disable bias modeling and save a
little time but perhaps be a little less accurate.

## Running regression / differential expression

Polee models are implemented in TensorFlow. You should have the the
tensorflow python library installed. They are faster with GPU, but this isn't
at all necessary.

### Writing experiment specification

To give models a (potentially large) list of samples along with accompanying
metadata, and YAML specification file is used. Here's an example I used with
a few samples from GTEx.

```yaml
samples:
  - name: brain_hippocampus-33
    factors:
      tissue: brain_hippocampus
    file: polee/SRR660969.prep.h5
  - name: brain_hippocampus-187
    factors:
      tissue: brain_hippocampus
    file: polee/SRR1475803.prep.h5
  - name: brain_hippocampus-71
    factors:
      tissue: brain_hippocampus
    file: polee/SRR1312743.prep.h5
  - name: brain_hippocampus-29
    factors:
      tissue: brain_hippocampus
    file: polee/SRR660733.prep.h5
  - name: brain_amygdala-113
    factors:
      tissue: brain_amygdala
    file: polee/SRR1407134.prep.h5
  - name: brain_amygdala-1
    factors:
      tissue: brain_amygdala
    file: polee/SRR598124.prep.h5
```

Important features: there must be a top-level `samples` entry with a list of
samples. Each sample has a `file` entry pointing to the file that the
`prep-sample` command produced. Each sample also a has a `name`, and list of
named `factors` that we might want to regress over.

### Calling transcript differential expression

With samples prepared and listed in a YAML file (let's call it
`experiment.yml`), we can detect differential expression.

```sh
polee model regression \
    --factors tissue \
    --nonredundant \
    --output regression.csv \
    experiment.yml
```

The `--factors` argument accepts a comma-separated list of factors (which
must be present in the YAML file) to include in the regression.

By default, regression assumes a global mean expression, and tries to detect
deviation from that. The `--nonredundant` option changes this so that
deviation is detected between the two tissues, which is a more common
approach when there are only two conditions being compared.

The result of this will a be a large csv file giving the results of the
results of the regression.

```csv
factor,transcript_id,min_effect_size,post_mean_effect,lower_credible,upper_credible
tissue:brain_amygdala,transcript:ENST00000456328,0.023486,-0.022711,-0.435812,0.390389
tissue:brain_amygdala,transcript:ENST00000450305,0.024105,0.005090,-0.420567,0.430748
tissue:brain_amygdala,transcript:ENST00000488147,0.025453,-0.097340,-0.482681,0.288001
...
```

The model here is Bayesian linear regression on log expression values. Unlike
null-hypothesis testing, we are interested in assessing probability of effect
sizes. The main way significance is assessed here is with "minimum effect
size" (the `min_effect_size`) column. "Effect size" here is the log2
fold-change. The minimum effect size is a number δ where there is a 90%
probability that the true effect size is greater than δ. The 90% here is the
default confidence, but it can be modified with the
`--min-effect-size-coverage` option. For example, with
`--min-effect-size-coverage 0.25` the results will report a minimum effect
size at 75%.

The `lower_credible` and `upper_credible` give a credible interval on the log2 fold-change. By default this is a 95% credible interval, but can be changes with
the `--lower-credible` and `--upper-credible` options.

### Other features

Gene differential expression and regression can also be run by passing the
`--feature gene` option. Somewhat experimental, there are also regression
models to look at splicing in particular. The `gene-isoform` features does
simultaneous regression over gene expression and isoform mixture, and
`splice-feature` will look specifically at differential splicing features
like cassette exons. More documentation on these methods will be forthcoming.

# Interoperating with sleuth

Approximate likelihood can be used with sleuth, which can give more accurate
transcript differential expression calls. (Gene differential expression calls
seem to be somewhat worse, likely due sleuth filtering heuristics. This is
something we're investigating.)

To use with sleuth, we generate samples from the approximate likelihood to
mimic kallisto bootstrap samples. To do this, use the `sample` command.

```sh
polee sample --num-samples 100 --kallisto -o abundance.h5 prepared-sample.h5
```

Now `abundance.h5` can be used as if it were kallisto output.
