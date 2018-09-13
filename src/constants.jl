
# various parameters than are generally fixed


# Whether to use a prior that encodes the assumption that most transcripts have
# low expression. Set to false by the right command line option.
const INFORMATIVE_PRIOR = true


# Increment this whenever we introduce a non-backward compatible change to
# the approximated likelihood function or its serialization format.
const PREPARED_SAMPLE_FORMAT_VERSION = 1


# mask of flags we care about
const USED_BAM_FLAGS =
    0x001 | # paired-end
    0x002 | # all pairs aligned
    0x004 | # unmapped
    0x010 | # reverse complement
    0x040 | # read 1
    0x080   # read 2

# For single-end sequencing, make some assumptions about fragment length distribution
const FALLBACK_FRAGLEN_MEAN = 150
const FALLBACK_FRAGLEN_SD  = 50

# Disallow fragments longer than this
const MAX_FRAG_LEN = 1500

# Don't use emperical distribution if fewer than this many paired-end reads
const MIN_FRAG_LEN_COUNT = 1000

# Effective lengths are set to max(ef, MIN_EFFECTIVE_LENGTH). Very small
# effective lengths can give very high and probably erroneous expression estimates.
const MIN_EFFECTIVE_LENGTH = 1.0f0

# Fragment is considered incompatible with a transcript if it has a conditional
# probability less than this.
const MIN_FRAG_PROB = 1e-12

# Epsilon used to clamp the y variable during likelihood approximation
const LIKAP_Y_EPS = 1e-10


# Constants used for optimization with ADAM
const ADAM_INITIAL_LEARNING_RATE = 1.0
const ADAM_LEARNING_RATE_DECAY = 2e-2
const ADAM_EPS = 1e-8
const ADAM_RV = 0.9
const ADAM_RM = 0.7


const LIKAP_NUM_STEPS = 400
const LIKAP_NUM_MC_SAMPLES = 5


# Parameters to InverseGamma priors on variance parameters
# const SIGMA_ALPHA0 = 0.1
# const SIGMA_BETA0  = 10.0

# Note: tensorflow parameterizes inverse gamma by recipricals because its a fucking sadist
const SIGMA_ALPHA0 = 0.001f0
const SIGMA_BETA0  = 0.001f0

# amount of sequence up- and downstream of the fragment sequence to include.
const BIAS_SEQ_INNER_CTX  = 15
const BIAS_SEQ_OUTER_CTX  = 5

# Number of bins to divide sequence fragments into for counting nucleotide frequencies
const BIAS_NUM_FREQ_BINS = 4

# Number of fragment lengths to sum over when estimating effective length
# (larger number is more accurate but slower)
const BIAS_EFFLEN_NUM_FRAGLENS = 200

