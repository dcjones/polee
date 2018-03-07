
# various parameters than are generally fixed

# mask of flags we care about
const USED_BAM_FLAGS =
    0x001 | # paired-end
    0x002 | # all pairs aligned
    0x004 | # unmapped
    0x010 | # reverse complement
    0x040 | # read 1
    0x080   # read 2

# Disallow fragments longer than this
const MAX_FRAG_LEN = 1500

# Don't use emperical distribution if fewer than this many paired-end reads
const MIN_FRAG_LEN_COUNT = 1000

# Effective lengths are set to max(ef, MIN_EFFECTIVE_LENGTH). Very small
# effective lengths can give very high and probably erroneous expression estimates.
const MIN_EFFECTIVE_LENGTH = 10.0f0

# Fragment is considered incompatible with a transcript if it has a conditional
# probability less than this.
const MIN_FRAG_PROB = 1e-10

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
const SIGMA_ALPHA0 = 10.0f0
const SIGMA_BETA0  = 0.1f0
