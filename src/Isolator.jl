
module Isolator

using ProgressMeter
using Bio.Intervals
using Bio.StringFields
using Bio.Align

include("hattrie.jl")
include("transcripts.jl")
include("reads.jl")

rs = Reads("1yr-1.bam")
ts = Transcripts("/home/dcjones/data/homo_sapiens/Homo_sapiens.GRCh38.85.gff3")

end


