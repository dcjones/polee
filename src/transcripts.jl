

struct Exon
    first::Int64
    last::Int64
end


function Base.length(e::Exon)
    return e.last - e.first + 1
end


function Base.isless(a::Exon, b::Exon)
    return a.first < b.first
end


function Base.isless(a::Interval, b::Exon)
    return a.first < b.first
end


mutable struct TranscriptMetadata
    name::String
    id::Int
    exons::Vector{Exon}
    seq::Vector{DNA}
    left_bias::Vector{Float32}
    right_bias::Vector{Float32}

    function TranscriptMetadata(name, id)
        return new(name, id, Exon[], DNA[], Float32[], Float32[])
    end

    function TranscriptMetadata(name, id, exons, seq)
        return new(name, id, exons, seq, Float32[], Float32[])
    end
end


const Transcript = Interval{TranscriptMetadata}


function exonic_length(t::Transcript)
    el = 0
    for exon in t.metadata.exons
        el += length(exon)
    end
    return el
end


function Base.push!(t::Transcript, e::Exon)
    push!(t.metadata.exons, e)
    t.first = min(t.first, e.first)
    t.last = max(t.last, e.last)
    return e
end


const Transcripts = IntervalCollection{TranscriptMetadata}


mutable struct TranscriptsMetadata
    filename::String
    gffsize::Int
    gffhash::Vector{UInt8}
    excluded_transcripts_hash::Vector{UInt8}

    # kind indexed by transcript_id
    transcript_kind::Dict{String, String}

    # gene_id indexed by transcript_id
    gene_id::Dict{String, String}

    # gene info indexed by gene_id
    gene_name::Dict{String, String}
    gene_biotype::Dict{String, String}
    gene_description::Dict{String, String}
end


function TranscriptsMetadata()
    return TranscriptsMetadata(
        "", 0, UInt8[], UInt8[],
        Dict{String, String}(),
        Dict{String, String}(),
        Dict{String, String}(),
        Dict{String, String}(),
        Dict{String, String}())
end


function Base.haskey(record::GFF3.Record, key::String)
    for (i, k) in enumerate(record.attribute_keys)
        if GFF3.isequaldata(key, record.data, k)
            return true
        end
    end
    return false
end


"""
Get the first value corresponding to the key, or an empty string if the
attribute isn't present.
"""
function getfirst_else_empty(rec::GFF3.Record, key::String)
    return haskey(rec, key) ? GFF3.attributes(rec, key)[1] : ""
end


function Transcripts(filename::String, excluded_transcripts::Set{String}=Set{String}())
    prog_step = 1000
    prog = Progress(filesize(filename), 0.25, "Reading GFF3 file ", 60)

    reader = open(GFF3.Reader, filename)
    entry = eltype(reader)()

    transcript_id_by_name = Dict{String, Int}()
    transcript_by_id = Transcript[]
    metadata = TranscriptsMetadata()
    interned_seqnames = Dict{String, String}()

    i = 0
    count = 0
    # while !isnull(tryread!(reader, entry))
    while !eof(reader)
        try
            read!(reader, entry)
        catch ex
            if isa(ex, EOFError)
                break
            end
        end

        if (i += 1) % prog_step == 0
            ProgressMeter.update!(prog, position(reader.state.stream.source))
        end
        count += 1

        typ = GFF3.featuretype(entry)

        if typ == "exon"
            parent_name = getfirst_else_empty(entry, "Parent")
            if !isempty(excluded_transcripts) &&
               (startswith(parent_name, "transcript:") &&
                replace(parent_name, "transcript:" => "") ∈ excluded_transcripts) ||
               parent_name ∈ excluded_transcripts
                continue
            end

            if parent_name == ""
                error("Exon has no parent")
            end

            id = get!(transcript_id_by_name, parent_name,
                    length(transcript_id_by_name) + 1)

            if id > length(transcript_by_id)
                seqname = GFF3.seqid(entry)
                push!(transcript_by_id,
                    Transcript(get!(interned_seqnames, seqname, seqname),
                               GFF3.seqstart(entry), GFF3.seqend(entry),
                               GFF3.hasstrand(entry) ? GFF3.strand(entry) : STRAND_BOTH,
                               TranscriptMetadata(parent_name, id)))
            end
            push!(transcript_by_id[id].metadata.exons,
                Exon(GFF3.seqstart(entry), GFF3.seqend(entry)))
        elseif typ == "gene"
            # gene some gene metadata if it's available
            entry_id = getfirst_else_empty(entry, "ID")
            metadata.gene_name[entry_id]        = getfirst_else_empty(entry, "Name")
            metadata.gene_biotype[entry_id]     = getfirst_else_empty(entry, "biotype")
            metadata.gene_description[entry_id] = getfirst_else_empty(entry, "description")
        elseif typ != "CDS"
            # If this entry is neither a gene nor an exon, assume it's some
            # sort of transcript entry
            entry_id = getfirst_else_empty(entry, "ID")
            parent_name = getfirst_else_empty(entry, "Parent")
            if isempty(parent_name)
                parent_name = getfirst_else_empty(entry, "geneID")
            end
            metadata.gene_id[entry_id] = parent_name
            metadata.transcript_kind[entry_id] = typ
        end
    end

    for t in transcript_by_id
        sort!(t.metadata.exons)
    end

    # fix transcript start/last
    for (i, t) in enumerate(transcript_by_id)
        sort!(t.metadata.exons)
        transcript_by_id[i] = Transcript(t.seqname, t.metadata.exons[1].first,
                                         t.metadata.exons[end].last, t.strand, t.metadata)
    end

    finish!(prog)
    println("Read ", length(transcript_by_id), " transcripts")
    transcripts = IntervalCollection(transcript_by_id, true)

    # reassign transcript indexes to group by position
    # (since it can give the sparse matrix a somewhat better structure)
    for (tid, t) in enumerate(transcripts)
        t.metadata.id = tid
    end

    metadata.filename = filename
    metadata.gffsize = filesize(filename)
    metadata.gffhash = SHA.sha1(open(filename))
    metadata.excluded_transcripts_hash = SHA.sha1(join(",", excluded_transcripts))

    return transcripts, metadata
end


struct ExonIntron
    first::Int
    last::Int
    isexon::Bool
end


function Base.length(e::ExonIntron)
    return e.last - e.first + 1
end


"""
Iterate over exons and introns in a transcript in order Interval{Bool} where
metadata flag is true for exons.
"""
struct ExonIntronIter
    t::Transcript
end


function Base.iterate(it::ExonIntronIter)
    return iterate(it, (1, true))
end


@inline function Base.iterate(it::ExonIntronIter, state::Tuple{Int, Bool})
    i, isexon = state

    if state[1] == length(it.t.metadata.exons) && !state[2]
        return nothing
    end

    if isexon
        ex = it.t.metadata.exons[i]
        return ExonIntron(ex.first, ex.last, true), (i, false)
    else
        return ExonIntron(it.t.metadata.exons[i].last+1,
                          it.t.metadata.exons[i+1].first-1, false), (i+1, true)
    end
end


"""
Fragment length assuming the alignment pair read was derived from the given
transcript.

Return null if the alignment is not compatible with the transcript.
"""
function fragmentlength(t::Transcript, rs::Reads, alnpr::AlignmentPair)
    # allow matches overhanging into introns by <= this amount
    max_allowable_encroachment = 2

    # rule out obviously incompatible alignments
    if alnpr.first < t.first || alnpr.last > t.last
        return nothing
    end

    # set a1, a2 as leftmost and rightmost alignments
    a1 = nothing
    a2 = nothing
    if alnpr.metadata.mate1_idx > 0 && alnpr.metadata.mate2_idx > 0
        mate1 = rs.alignments[alnpr.metadata.mate1_idx]
        mate2 = rs.alignments[alnpr.metadata.mate2_idx]
        if mate1.leftpos <= mate2.leftpos
            a1, a2 = mate1, mate2
        else
            a1, a2 = mate2, mate1
        end
    elseif alnpr.metadata.mate1_idx > 0
        a1 = rs.alignments[alnpr.metadata.mate1_idx]
    else
        a1 = rs.alignments[alnpr.metadata.mate2_idx]
    end

    c1_iter = CigarIter(rs, a1)
    c1_next = iterate(c1_iter)

    exons = t.metadata.exons
    first_exon_idx = searchsortedlast(exons, alnpr)
    e1_idx = first_exon_idx
    e1_isexon = true
    e1_first, e1_last = exons[e1_idx].first, exons[e1_idx].last

    # amount of intronic dna spanned by the fragment
    intronlen = 0

    # skip any leading soft clipping
    if c1_next !== nothing && c1_next[1].op == OP_SOFT_CLIP
        c1_next = iterate(c1_iter, c1_next[2])
    end

    while e1_idx <= length(exons) && c1_next !== nothing
        c, c1_state = c1_next

        # case 1: e entirely precedes
        if e1_last < c.first
            @fragmentlength_next_exonintron!(exons, e1_idx, e1_isexon,
                                             e1_first, e1_last)

        # case 2: c is contained within e
        elseif c.last >= e1_first && c.last <= e1_last && c.first >= e1_first
            if e1_isexon
                if !is_exon_compatible(c.op)
                    return nothing
                end
            else
                if !is_intron_compatible(c.op)
                    return nothing
                end
                intronlen += e1_last - e1_first + 1
            end
            c1_next = iterate(c1_iter, c1_state)

        # case 3: soft clipping partiallly overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            c1_next = iterate(c1_iter, c1_state)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e1_last && c.op == OP_MATCH
            if e1_isexon && c.last - e1_last <= max_allowable_encroachment
                c1_next = CigarInterval(c.first, e1_last, c.op), c1_state
            elseif !e1_isexon && e1_last >= c.first &&
                   e1_last - c.first < max_allowable_encroachment
                c1_next = CigarInterval(e1_last + 1, c.last, c.op), c1_state
            else
                return nothing
            end

        # case 5: c precedes and partially overlaps e
        else
            return nothing
        end
    end

    if c1_next !== nothing
        return nothing
    end

    # alignment is compatible, but single-ended
    if a2 === nothing
        return 0
    end

    e2_sup_e1 = false # marks with e2 > e1

    c2_iter = CigarIter(rs, a2)
    c2_next = iterate(c2_iter)

    e2_idx = first_exon_idx
    e2_isexon = true
    e2_first, e2_last = exons[e2_idx].first, exons[e2_idx].last

    while e2_idx <= length(exons) && c2_next !== nothing
        c, c2_state = c2_next

        # case 1: e entirely precedes c
        if e2_last < c.first
            if !e2_isexon && e2_sup_e1
                intronlen += e2_last - e2_first + 1
            end

            if e1_idx <= length(exons) &&
               e1_first == e2_first &&
               e1_last == e2_last
                e2_sup_e1 = true
            end

            @fragmentlength_next_exonintron!(exons, e2_idx, e2_isexon,
                                             e2_first, e2_last)

        # case 2: c is contained within e
        elseif c.last >= e2_first && c.last <= e2_last && c.first >= e2_first
            if e2_isexon
                if !is_exon_compatible(c.op)
                    return nothing
                end
            else
                if !is_intron_compatible(c.op)
                    return nothing
                end
            end
            c2_next = iterate(c2_iter, c2_state)

        # case 3: soft clipping partially overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            c2_next = iterate(c2_iter, c2_state)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e2_last && c.op == OP_MATCH
            if e2_isexon && c.last - e2_last <= max_allowable_encroachment
                c2_next = CigarInterval(c.first, e2_last, c.op), c2_state
            elseif !e2_isexon && e2_last >= c.first &&
                e2_last - c.first < max_allowable_encroachment
                c2_next = CigarInterval(e2_last + 1, c.last, c.op), c2_state
            else
                return nothing
            end

        # case 5: c precedes and partially overlaps e
        else
            return nothing
        end
    end

    # skip any trailing soft clipping
    if c2_next !== nothing && c2_next[1].op == OP_SOFT_CLIP
        c2_next = iterate(c2_iter, c2_next[2])
    end

    if c2_next !== nothing
        return nothing
    end

    fraglen = max(a1.rightpos, a2.rightpos) -
              min(a1.leftpos, a2.leftpos) + 1 - intronlen

    if fraglen > 0
        return Int(fraglen)
    else
        return nothing
    end
end


"""
Return the fragments interval in transcript coordinates.
"""
function genomic_to_transcriptomic(
        t::Transcript, rs::Reads, alnpr::AlignmentPair, fraglen_median::Int=0)

    tlen = length(t.metadata.seq)

    fraglen = fragmentlength(t, rs, alnpr)
    if fraglen === nothing
        # incompatible fragment
        return 1:0
    end

    if fraglen <= 0
        fraglen = fraglen_median
        if fraglen <= 0
            return 1:0
        end
    end

    # set tpos to 5' most position of the fragment (relative to transcript)
    if alnpr.metadata.mate1_idx > 0 && alnpr.metadata.mate2_idx > 0
        aln1 = rs.alignments[alnpr.metadata.mate1_idx]
        aln2 = rs.alignments[alnpr.metadata.mate2_idx]
        if t.strand == STRAND_POS
            gpos = min(aln1.leftpos, aln2.leftpos)
        else
            gpos = max(aln1.rightpos, aln2.rightpos)
        end
        tpos = genomic_to_transcriptomic(t, gpos)
    else
        # single-strand where we may have to guess
        aln = alnpr.metadata.mate1_idx > 0 ?
            rs.alignments[alnpr.metadata.mate1_idx] :
            rs.alignments[alnpr.metadata.mate2_idx]

        alnstrand = aln.flag & SAM.FLAG_REVERSE != 0 ?
            STRAND_NEG : STRAND_POS

        if t.strand == STRAND_POS
            if alnstrand == STRAND_POS
                tpos = genomic_to_transcriptomic(t, aln.leftpos)
            else
                tpos = genomic_to_transcriptomic(t, aln.rightpos) - fraglen
            end
        else
            if alnstrand == STRAND_POS
                tpos = genomic_to_transcriptomic(t, aln.leftpos) - fraglen
            else
                tpos = genomic_to_transcriptomic(t, aln.rightpos)
            end
        end
    end

    # nudge reads that overhang (due to soft-clipping typically)
    if tpos <= 0
        fraglen += tpos - 1
        tpos = 1
    end

    if tpos+fraglen-1 > tlen
        fraglen = tlen - tpos + 1
    end

    fragint = Int(tpos):Int(tpos+fraglen-1)

    return Int(tpos):Int(tpos+fraglen-1)
end


function genomic_to_transcriptomic(t::Transcript, position::Integer)
    exons = t.metadata.exons
    i = searchsortedlast(exons, Exon(position, position))
    if i == 0 || exons[i].last < position
        # @show position
        # @show exons
        return 0
    else
        tpos = 1
        for j in 1:i-1
            tpos += exons[j].last - exons[j].first + 1
        end
        tpos += position - t.metadata.exons[i].first
        if t.strand == STRAND_NEG
            tpos = exonic_length(t) - tpos + 1
        end
        return tpos
    end
end


"""
Return a new IntervalCollection containing introns, with metadata containing
transcripts id for each transcript including that intron.
"""
function get_introns(ts::Transcripts)
    introns = IntervalCollection{Vector{Int}}()
    for t in ts
        for i in 2:length(t.metadata.exons)
            first = t.metadata.exons[i-1].last + 1
            last  = t.metadata.exons[i].first - 1

            key = Interval{Nothing}(t.seqname, first, last, t.strand, nothing)
            entry = findfirst(introns, key, filter=(a,b)->a.strand==b.strand)
            if entry === nothing
                entry = Interval{Vector{Int}}(t.seqname, first, last, t.strand, Int[t.metadata.id])
                push!(introns, entry)
            else
                push!(entry.metadata, t.metadata.id)
            end
        end
    end

    return introns
end


function get_cassette_exons(ts::Transcripts)
    get_cassette_exons(ts, get_introns(ts))
end


function get_cassette_exons(ts::Transcripts, introns::IntervalCollection{Vector{Int}})
    function match_strand_exon(a, b)
        return a.strand == b.strand &&
               a.metadata[1] == b.metadata[1] &&
               a.metadata[2] == b.metadata[2]
    end

    # build a set representing introns flanking internal exons
    flanking_introns = IntervalCollection{Tuple{Int, Int, Vector{Int}}}()
    for t in ts
        i = 3
        while i <= length(t.metadata.exons)
            e1 = t.metadata.exons[i-2]
            e2 = t.metadata.exons[i-1]
            e3 = t.metadata.exons[i]
            key = Interval{Tuple{Int,Int}}(t.seqname, e1.last+1, e3.first-1, t.strand,
                                           (e2.first, e2.last))

            entry = findfirst(flanking_introns, key, filter=match_strand_exon)
            if entry === nothing
                push!(flanking_introns,
                      Interval{Tuple{Int, Int, Vector{Int}}}(
                          key.seqname, key.first, key.last, key.strand,
                          (key.metadata[1], key.metadata[2], Int[t.metadata.id])))
            else
                push!(entry.metadata[3], t.metadata.id)
            end

            i += 1
        end
    end

    # find cassette exons
    cassette_exons = Tuple{Interval{Vector{Int}}, Interval{Tuple{Int, Int, Vector{Int}}}}[]
    for flanks in flanking_introns
        intron = findfirst(introns, flanks, filter=(a,b)->a.strand==b.strand)
        if intron !== nothing
            push!(cassette_exons, (intron, flanks))
        end
    end

    # find mutually exclusive exons
    exons_by_flanks = Dict{Interval{Nothing}, Vector{Tuple{Int, Int, Vector{Int}}}}()
    for flanks in flanking_introns
        key = Interval{Nothing}(flanks.seqname, flanks.first, flanks.last, flanks.strand, nothing)
        if !haskey(exons_by_flanks, key)
            exons_by_flanks[key] = Tuple{Int, Int, Vector{Int}}[]
        end
        push!(exons_by_flanks[key], flanks.metadata)
    end

    mutex_exons = Tuple{Interval{Vector{Int}}, Interval{Vector{Int}}}[]
    for (flanks, exons) in exons_by_flanks
        if length(exons) <= 1
            continue
        end
        sort!(exons)

        # collapse exons into unions
        flattened_exons = Tuple{Int, Int, Vector{Int}}[]
        for exon in exons
            first, last, tids = exon
            if isempty(flattened_exons) || first > flattened_exons[end][2]
                push!(flattened_exons, exon)
            else
                flattened_exons[end] =
                    (min(flattened_exons[end][1], first),
                     max(flattened_exons[end][2], last),
                     vcat(flattened_exons[end][3], tids))
            end
        end

        if length(flattened_exons) <= 1
            continue
        elseif length(flattened_exons) == 2
            first_a, last_a, tids_a = flattened_exons[1]
            first_b, last_b, tids_b = flattened_exons[2]

            push!(mutex_exons,
                (Interval(flanks.seqname, first_a, last_a, flanks.strand, tids_a),
                Interval(flanks.seqname, first_b, last_b, flanks.strand, tids_b)))
        else
            # A weird multiple (>2) mutually exclusive splicing event, which
            # we'll ignore for now, but possibly we should try to consider.
            # E.g. there are a few dozen of these events in mouse.
            # @show flanks
        end
    end

    return cassette_exons, mutex_exons
end


function get_alt_donor_acceptor_sites(ts::Transcripts)

    function match_strand(a, b)
        return a.strand == b.strand
    end

    # hold the exons transcript number along with intron flanks, if it has them.
    IntronFlanks = Tuple{Int, Nullable{Int}, Nullable{Int}}

    exons = IntervalCollection{IntronFlanks}()
    for t in ts
        for i in 1:length(t.metadata.exons)
            exon = Interval{IntronFlanks}(
                t.seqname, t.metadata.exons[i].first, t.metadata.exons[i].last,
                t.strand,
                (t.metadata.id,
                 i == 1 ?
                     Nullable{Int}() : Nullable{Int}(t.metadata.exons[i-1].last+1),
                 i == length(t.metadata.exons) ?
                     Nullable{Int}() : Nullable{Int}(t.metadata.exons[i+1].first-1)))

            push!(exons, exon)
        end
    end

    # interval contains the shorter intron. First two metadata fields contain
    # the longer intron. Second two fields give the transcript numbers for those
    # using the shorter and longer introns respectively.
    AltAccDonMetadata = Tuple{Int, Int, Set{Int}, Set{Int}}
    alt_accdon_sites = IntervalCollection{AltAccDonMetadata}()

    RetIntronMetadata = Tuple{Set{Int}, Set{Int}}
    retained_introns = IntervalCollection{RetIntronMetadata}()

    for (a, b) in eachoverlap(exons, exons, filter=match_strand)
        # skip exons without flanks for now
        if isnull(a.metadata[2]) || isnull(a.metadata[3]) ||
           isnull(b.metadata[2]) || isnull(b.metadata[3])
            continue
        end

        has_alt_accdon = false
        has_retained_intron = false

        # alternate acceptor/donor site case 1
        #
        #    ...aaaaa--------------|
        #    ...bbbbbbb------------|
        # or
        #    ...aaaaaaa------------|
        #    ...bbbbb--------------|
        #
        if get(a.metadata[3]) == get(b.metadata[3]) && a.last != b.last
            short_last = long_last = get(a.metadata[3])
            if a.last < b.last
                short_first, long_first = a.last + 1,    b.last + 1
                short_tid,   long_tid   = a.metadata[1], b.metadata[1]
            else
                long_first, short_first = a.last + 1,    b.last + 1
                long_tid,   short_tid   = a.metadata[1], b.metadata[1]
            end
            has_alt_accdon = true

        # alternate acceptor/donor site case 2
        #
        #    |--------------aaaaa...
        #    |------------bbbbbbb...
        # or
        #    |--------------bbbbb...
        #    |------------aaaaaaa...
        #
        elseif get(a.metadata[2]) == get(b.metadata[2]) && a.first != b.first
            short_first = long_first = get(a.metadata[2])
            if a.first > b.first
                short_last, long_last = b.first - 1,   a.first - 1
                short_tid,  long_tid  = b.metadata[1], a.metadata[1]
            else
                long_last, short_last = b.first - 1,   a.first - 1
                long_tid,  short_tid  = b.metadata[1], a.metadata[1]
            end
            has_alt_accdon = true

        # retained intron case 1
        #
        #    aaaaa-------|
        #    bbbbbbbbbbbbbbb...
        # or
        #    bbbbb-------|
        #    aaaaaaaaaaaaaaa...
        elseif get(a.metadata[3]) < b.last
            retained_intron_first = a.last + 1
            retained_intron_last  = get(a.metadata[3])
            retained_intron_include_tid = b.metadata[1]
            retained_intron_exclude_tid = a.metadata[1]
            has_retained_intron = true
        elseif get(b.metadata[3]) < a.last
            retained_intron_first = b.last + 1
            retained_intron_last  = get(b.metadata[3])
            retained_intron_include_tid = a.metadata[1]
            retained_intron_exclude_tid = b.metadata[1]
            has_retained_intron = true

        # retained intron case 1
        #
        #     |-------aaaaa
        #    bbbbbbbbbbbbbbb
        # or
        #     |-------bbbbb
        #    aaaaaaaaaaaaaaa
        elseif get(a.metadata[2]) > b.first
            retained_intron_first = get(a.metadata[2])
            retained_intron_last  = a.first - 1
            retained_intron_include_tid = b.metadata[1]
            retained_intron_exclude_tid = a.metadata[1]
            has_retained_intron = true
        elseif get(b.metadata[2]) > a.last
            retained_intron_first = get(b.metadata[2])
            retained_intron_last  = b.first - 1
            retained_intron_include_tid = a.metadata[1]
            retained_intron_exclude_tid = b.metadata[1]
            has_retained_intron = true
        end

        if has_alt_accdon
            key = Interval{Tuple{Int, Int}}(
                a.seqname, short_first, short_last, a.strand, (long_first, long_last))
            entry = findfirst(alt_accdon_sites, key,
                filter=(a,b) -> a.strand == b.strand &&
                    a.metadata[1] == b.metadata[1] &&
                    a.metadata[2] == b.metadata[2])
            @assert short_tid != long_tid
            if entry === nothing
                entry = Interval{AltAccDonMetadata}(
                    a.seqname, short_first, short_last, a.strand,
                    (Int(long_first), Int(long_last), Set{Int}(short_tid), Set{Int}(long_tid)))
                push!(alt_accdon_sites, entry)
            else
                push!(entry.metadata[3], short_tid)
                push!(entry.metadata[4], long_tid)
            end
        elseif has_retained_intron
            key = Interval{Nothing}(
                a.seqname, retained_intron_first, retained_intron_last,
                a.strand, nothing)
            entry = findfirst(retained_introns, key,
                filter=(a,b) -> a.strand == b.strand)
            @assert retained_intron_exclude_tid != retained_intron_include_tid
            if entry === nothing
                entry = Interval{RetIntronMetadata}(
                    a.seqname, retained_intron_first, retained_intron_last,
                    a.strand, (
                        Set{Int}(retained_intron_include_tid),
                        Set{Int}(retained_intron_exclude_tid)))
                push!(retained_introns, entry)
            else
                push!(entry.metadata[1], retained_intron_include_tid)
                push!(entry.metadata[2], retained_intron_exclude_tid)
            end
        end
    end

    return alt_accdon_sites, retained_introns
end


"""
Merge values x in xs into intervals containing values that are within d of each other.
"""
function merge_adjacent(xs::Vector{T}, d) where {T}
    sort!(xs)
    ys = Vector{T}[T[xs[1]]]
    for i in 2:length(xs)
        if xs[i] - ys[end][end] <= d
            push!(ys[end], xs[i])
        else
            push!(ys, T[xs[i]])
        end
    end
    return ys
end


"""
Find alternative five-prime and three-prime transcript ends.
"""
function get_alt_fp_tp_ends(
        ts::Transcripts, ts_metadata::TranscriptsMetadata, merge_distance=250)


    # TODO: ensembl tends to give pretty much every transcript a different start
    # and ends site, to do really do this, we should do some sort of clustering
    # of stards/ends.

    transcripts_by_gene_id = Dict{String, Vector{Transcript}}()
    for t in ts
        gene_id = ts_metadata.gene_id[t.metadata.name]
        if !haskey(transcripts_by_gene_id, gene_id)
            transcripts_by_gene_id[gene_id] = Transcript[]
        end
        push!(transcripts_by_gene_id[gene_id], t)
    end

    alt_fp_ends = Interval{Tuple{Int, Int, Vector{Int}, Vector{Int}}}[]
    alt_tp_ends = Interval{Tuple{Int, Int, Vector{Int}, Vector{Int}}}[]

    for (gene_id, ts) in transcripts_by_gene_id
        if length(ts) <= 1
            continue
        end

        firsts_set = Set{Int}()
        lasts_set = Set{Int}()

        for t in ts
            push!(firsts_set, t.metadata.exons[1].first)
            push!(lasts_set, t.metadata.exons[end].last)
        end

        min_first = minimum(firsts_set)
        max_last = minimum(lasts_set)

        # merge nearby firsts and lasts
        firsts = merge_adjacent(collect(firsts_set), merge_distance)
        lasts = merge_adjacent(collect(lasts_set), merge_distance)

        # consider alternative firsts
        if length(firsts) > 1
            for first_cluster in firsts
                inc_ids = Int[]
                exc_ids = Int[]
                for t in ts
                    if in(t.metadata.exons[1].first, first_cluster)
                        push!(inc_ids, t.metadata.id)
                    else
                        push!(exc_ids, t.metadata.id)
                    end
                end

                entry = Interval(
                    ts[1].seqname, first_cluster[1], first_cluster[end],
                    ts[1].strand, (min_first, max_last, inc_ids, exc_ids))
                if ts[1].strand == STRAND_POS
                    push!(alt_fp_ends, entry)
                else
                    push!(alt_tp_ends, entry)
                end

                # don't bother include the reciprocal
                if length(firsts) == 2
                    break
                end
            end
        end

        # consider alternative ends
        if length(lasts) > 1
            for last_cluster in lasts
                inc_ids = Int[]
                exc_ids = Int[]
                for t in ts
                    if in(t.metadata.exons[end].last, last_cluster)
                        push!(inc_ids, t.metadata.id)
                    else
                        push!(exc_ids, t.metadata.id)
                    end
                end

                entry = Interval(
                    ts[1].seqname, last_cluster[1], last_cluster[end],
                    ts[1].strand, (min_first, max_last, inc_ids, exc_ids))
                if ts[1].strand == STRAND_POS
                    push!(alt_tp_ends, entry)
                else
                    push!(alt_fp_ends, entry)
                end

                # don't bother include the reciprocal
                if length(lasts) == 2
                    break
                end
            end
        end
    end

    return alt_fp_ends, alt_tp_ends
end


"""
Serialize a GFF3 file into sqlite3 database.
"""
function write_transcripts(output_filename, transcripts, metadata)
    db = SQLite.DB(output_filename)

    # Gene Table
    # ----------

    gene_nums = Dict{String, Int}()
    for (transcript_id, gene_id) in metadata.gene_id
        get!(gene_nums, gene_id, length(gene_nums) + 1)
    end

    SQLite.execute!(db, "drop table if exists genes")
    SQLite.execute!(db,
        """
        create table genes
        (
            gene_num INT PRIMARY KEY,
            gene_id TEXT,
            gene_name TEXT,
            gene_biotype TEXT,
            gene_description TEXT
        )
        """)

    ins_stmt = SQLite.Stmt(db, "insert into genes values (?1, ?2, ?3, ?4, ?5)")
    SQLite.execute!(db, "begin transaction")
    for (gene_id, gene_num) in gene_nums
        SQLite.bind!(ins_stmt, 1, gene_num)
        SQLite.bind!(ins_stmt, 2, gene_id)
        SQLite.bind!(ins_stmt, 3, get(metadata.gene_name, gene_id, ""))
        SQLite.bind!(ins_stmt, 4, get(metadata.gene_biotype, gene_id, ""))
        SQLite.bind!(ins_stmt, 5, get(metadata.gene_description, gene_id, ""))
        SQLite.execute!(ins_stmt)
    end
    SQLite.execute!(db, "end transaction")

    # Transcript Table
    # ----------------

    SQLite.execute!(db, "drop table if exists transcripts")
    SQLite.execute!(db,
        """
        create table transcripts
        (
            transcript_num INT PRIMARY KEY,
            transcript_id TEXT,
            kind TEXT,
            seqname TEXT,
            strand INT,
            gene_num INT
        )
        """)
    ins_stmt = SQLite.Stmt(db,
        "insert into transcripts values (?1, ?2, ?3, ?4, ?5, ?6)")
    SQLite.execute!(db, "begin transaction")
    for t in transcripts
        SQLite.bind!(ins_stmt, 1, t.metadata.id)
        SQLite.bind!(ins_stmt, 2, String(t.metadata.name))
        SQLite.bind!(ins_stmt, 3, get(metadata.transcript_kind, t.metadata.name, ""))
        SQLite.bind!(ins_stmt, 4, String(t.seqname))
        SQLite.bind!(ins_stmt, 5,
            t.strand == STRAND_POS ? 1 :
            t.strand == STRAND_NEG ? -1 : 0)
        SQLite.bind!(ins_stmt, 6, gene_nums[metadata.gene_id[t.metadata.name]])
        SQLite.execute!(ins_stmt)
    end
    SQLite.execute!(db, "end transaction")


    # Exon Table
    # ----------

    SQLite.execute!(db, "drop table if exists exons")
    SQLite.execute!(db,
        """
        create table exons
        (
            transcript_num INT,
            first INT,
            last INT
        )
        """)

    ins_stmt = SQLite.Stmt(db, "insert into exons values (?1, ?2, ?3)")
    SQLite.execute!(db, "begin transaction")
    for t in transcripts
        for exon in t.metadata.exons
            SQLite.bind!(ins_stmt, 1, t.metadata.id)
            SQLite.bind!(ins_stmt, 2, exon.first)
            SQLite.bind!(ins_stmt, 3, exon.last)
            SQLite.execute!(ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")

    return db
end


# TODO: Read from sqlite3


"""
Does two things:
    (1) throws an error if a transcript appears in more than one feature
    (2) pads the matrix with extra single transcript features to account
        for those not present in the given features.

Returns the number of extra features added.
"""
function regularize_disjoint_feature_matrix!(feature_idxs, transcript_idxs, n)

    present_transcripts = IntSet()
    for tid in transcript_idxs
        if tid ∈ present_transcripts
            error("Transcript $(tid) is part of multiple features.")
        end
        push!(present_transcripts, tid)
    end

    num_aux_features = 0
    if length(present_transcripts) < n
        for j in 1:n
            if j ∉ present_transcripts
                push!(feature_idxs, length(feature_idxs))
                push!(transcript_idxs, j)
                num_aux_features += 1
            end
        end
    end

    return num_aux_features
end


"""
Generate a m-by-n sparse 0/1 matrix F where m is the number of genes, and n
is the number of transcripts, such that givin transcript expression y, Fy
gives gene expression.
"""
function gene_feature_matrix(ts::Transcripts, ts_metadata::TranscriptsMetadata)
    # it's possible that there are genes in the metadata that have no
    # transcripts, if transcripts are being blacklisted
    used_genes = Set{String}()
    for t in ts
        push!(used_genes, ts_metadata.gene_id[t.metadata.name])
    end

    gene_nums = Dict{String, Int}()
    for (transcript_id, gene_id) in ts_metadata.gene_id
        if gene_id ∈ used_genes
            get!(gene_nums, gene_id, length(gene_nums) + 1)
        end
    end

    I = Int[]
    J = Int[]
    for t in ts
        gene_num = gene_nums[ts_metadata.gene_id[t.metadata.name]]
        push!(I, gene_num)
        push!(J, t.metadata.id)
    end

    m = length(gene_nums)
    names = Array{String}(undef, m)
    ids = Array{String}(undef, m)
    for (gene_id, gene_num) in gene_nums
        ids[gene_num] = gene_id
        names[gene_num] = get(ts_metadata.gene_name, gene_id, "")
    end

    return (m, I, J, ids, names)
end



function splicing_feature_matrix(ts::Transcripts, ts_metadata::TranscriptsMetadata)
    # TODO:
end