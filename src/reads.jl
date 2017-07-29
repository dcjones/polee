

# Don't consider read pairs more than this far apart
const MAX_PAIR_DISTANCE = 500000

immutable Alignment
    id::UInt32
    refidx::Int32
    leftpos::Int32
    rightpos::Int32
    flag::UInt16
    # indexes into Reads.cigardata encoding alignment
    cigaridx::UnitRange{UInt32}
end


# SAM/BAM spec says pos should be the first matching position, which is hugely
# inconvenient. We compute the actual start position here.
function _leftposition(rec::BAM.Record)
    pos = leftposition(rec)
    offset = BAM.seqname_length(rec)
    for i in offset+1:4:offset+BAM.n_cigar_op(rec)*4
        x = unsafe_load(Ptr{UInt32}(pointer(rec.data, i)))
        op = Operation(x & 0x0f)
        if op != OP_MATCH
            pos -= x >> 4
        else
            break
        end
    end
    return pos
end


function _alignment_length(rec::BAM.Record)
    offset = BAM.seqname_length(rec)
    length::Int = 0
    for i in offset+1:4:offset+BAM.n_cigar_op(rec)*4
        x = unsafe_load(Ptr{UInt32}(pointer(rec.data, i)))
        op = Operation(x & 0x0f)
        if ismatchop(op) || isdeleteop(op) || op == OP_SOFT_CLIP
            length += x >> 4
        end
    end
    return length
end


function _rightposition(rec::BAM.Record)
    return Int32(_leftposition(rec) + _alignment_length(rec) - 1)
end


immutable AlignmentPairMetadata
    mate1_idx::UInt32
    mate2_idx::UInt32
end
const AlignmentPair = Interval{AlignmentPairMetadata}


function Base.isless(a::Alignment, b::Alignment)
    if a.refidx < b.refidx
        return true
    elseif a.refidx == b.refidx
        if a.id < b.id
            return true
        elseif a.id == b.id
            a_flag = a.flag & SAM.FLAG_READ2
            b_flag = b.flag & SAM.FLAG_READ2

            if a_flag == b_flag
                return a.leftpos < b.leftpos
            else
                return a_flag < b_flag
            end
        else
            return false
        end
    end
    return false
end


"""
Check that alignments are equal except for presense/absense of the secondary alignment flag.
"""
function isequiv(cigardata, a::Alignment, b::Alignment)
    if a.id != b.id  ||
       a.refidx != b.refidx ||
       a.leftpos != b.leftpos ||
       (a.flag & (~SAM.FLAG_SECONDARY)) != (b.flag & (~SAM.FLAG_SECONDARY)) ||
       length(a.cigaridx) != length(b.cigaridx)
        return false
    end

    for (i, j) in zip(a.cigaridx, b.cigaridx)
        if cigardata[i] != cigardata[j]
            return false
        end
    end

    return true
end


immutable Reads
    alignments::Vector{Alignment}
    alignment_pairs::IntervalCollection{AlignmentPairMetadata}
    cigardata::Vector{UInt32}
end


function cigar_from_ptr(data::Ptr{UInt32}, i)
    x = unsafe_load(data + i - 1)
    op = Operation(x & 0x0f)
    len = x >> 4
    return (op, len)
end


function Reads(filename::String, excluded_seqs::Set{String})
    if filename == "-"
        reader = BAM.Reader(STDIN)
        prog = Progress(filesize(filename), 0.25, "Reading BAM file ", 60)
        from_file = false
    else
        reader = open(BAM.Reader, filename)
        prog = Progress(0, 0.25, "Reading BAM file ", 60)
        from_file = true
    end
    return Reads(reader, prog, from_file, excluded_seqs)
end


function Reads(reader::BAM.Reader, prog::Progress, from_file::Bool,
               excluded_seqs::Set{String})
    prog_step = 1000
    entry = eltype(reader)()
    readnames = HATTrie()
    alignments = Alignment[]
    cigardata = UInt32[]

    # don't bother with reads from sequences with no transcripts
    excluded_refidxs = IntSet()
    for (refidx, seqname) in enumerate(reader.refseqnames)
        if seqname in excluded_seqs
            push!(excluded_refidxs, refidx)
        end
    end

    i = 0
    while !eof(reader)
        try
            read!(reader, entry)
        catch ex
            if isa(ex, EOFError)
                break
            end
        end

        if from_file && (i += 1) % prog_step == 0
            ProgressMeter.update!(prog, position(reader.stream.io))
        end

        if !BAM.ismapped(entry)
            continue
        end

        if entry.refid + 1 in excluded_refidxs
            continue
        end

        # copy cigar data over if there are any non-match operations
        cigarptr = Ptr{UInt32}(pointer(
                entry.data, 1 + BAM.seqname_length(entry)))
        cigarlen = BAM.n_cigar_op(entry)

        N = UInt32(length(cigardata))
        cigaridx = N+1:N
        if cigarlen > 1 || cigar_from_ptr(cigarptr, 1)[1] != OP_MATCH
            cigaridx = N+1:N+1+cigarlen-1
            resize!(cigardata, N + cigarlen)
            unsafe_copy!(pointer(cigardata, N + 1),
                         cigarptr, cigarlen)
        end

        id = get!(readnames, BioAlignments.seqname(entry), length(readnames) + 1)
        push!(alignments, Alignment(id, entry.refid + 1,
                                    _leftposition(entry), _rightposition(entry),
                                    BAM.flag(entry), cigaridx))
    end
    finish!(prog)

    @printf("Read %9d reads\nwith %9d alignments\n",
            length(readnames), length(alignments))

    # group alignments into alignment pair intervals
    @time sort!(alignments)

    tic()
    # first partition alignments by reference sequence
    blocks = Array{UnitRange{Int}}(0)
    i = 1
    while i <= length(alignments)
        j = i
        while j + 1 <= length(alignments) &&
              alignments[i].refidx == alignments[j].refidx
             j += 1
        end

        push!(blocks, i:j)
        i = j + 1
    end

    trees = Array{GenomicFeatures.ICTree{AlignmentPairMetadata}}(length(blocks))
    alignment_pairs = make_interval_collection(alignments, reader.refseqnames, blocks, trees, cigardata)

    return Reads(alignments, alignment_pairs, cigardata)
end


function make_interval_collection(alignments, seqnames, blocks, trees, cigardata)
    # Threads.@threads for blockidx in 1:length(blocks)
    for blockidx in 1:length(blocks)
        block = blocks[blockidx]
        seqname = seqnames[alignments[block.start].refidx]
        tree = GenomicFeatures.ICTree{AlignmentPairMetadata}()

        i, j = block.start, block.start
        while i <= block.stop
            j1 = i
            while j1 + 1 <= block.stop &&
                  alignments[i].id == alignments[j1+1].id &&
                  (alignments[j1+1].flag & SAM.FLAG_READ2) == 0
                j1 += 1
            end

            j2 = j1
            while j2 + 1 <= block.stop &&
                  alignments[i].id == alignments[j2+1].id
                j2 += 1
            end

            # now i:j1 are mate1 alignments, and j1+1:j2 are mate2 alignments

            # examine every potential mate1, mate2 pair
            alncnt = 0
            for k1 in i:j1
                m1 = alignments[k1]

                if k1 > i && isequiv(cigardata, m1, alignments[k1-1])
                    continue
                end

                for k2 in j1+1:j2
                    m2 = alignments[k2]

                    if k2 > j1+1 && isequiv(cigardata, m2, alignments[k2-1])
                        continue
                    end

                    minpos = min(m1.leftpos, m2.leftpos)
                    maxpos = max(m1.rightpos, m2.rightpos)

                    if maxpos - minpos > MAX_PAIR_DISTANCE
                        continue
                    end

                    strand = m1.flag & SAM.FLAG_REVERSE != 0 ?
                                STRAND_NEG : STRAND_POS

                    alnpr = AlignmentPair(
                        seqname, minpos, maxpos, strand,
                        AlignmentPairMetadata(k1, k2))
                    push!(tree, alnpr)
                end
            end

            # handle single-end reads
            if isempty(j1+1:j2)
                for k in i:j1
                    m = alignments[k]
                    if m.flag & SAM.FLAG_READ1 != 0
                        continue
                    end

                    minpos = m.leftpos
                    maxpos = m.rightpos

                    strand = m.flag & SAM.FLAG_REVERSE != 0 ?
                                STRAND_NEG : STRAND_POS

                    alnpr = AlignmentPair(
                        seqname, minpos, maxpos, strand,
                        AlignmentPairMetadata(k, 0))
                    push!(tree, alnpr)
                    alncnt += 1
                end
            end

            i = j2 + 1
        end

        trees[blockidx] = tree
    end

    # piece together the interval collection
    alignment_pairs = IntervalCollection{AlignmentPairMetadata}()
    treemap = Dict{String, GenomicFeatures.ICTree{AlignmentPairMetadata}}()
    iclength = 0
    for (block, tree) in zip(blocks, trees)
        seqname = seqnames[alignments[block.start].refidx]
        treemap[seqname] = tree
        iclength += length(tree)
    end

    alignment_pairs.trees = treemap
    alignment_pairs.length = iclength
    alignment_pairs.ordered_trees_outdated = true
    GenomicFeatures.update_ordered_trees!(alignment_pairs)

    return alignment_pairs
end


function cigar_len(aln::Alignment)
    return max(length(aln.cigaridx), 1)
end


immutable CigarInterval
    first::Int
    last::Int
    op::Operation
end


function Base.length(c::CigarInterval)
    return c.last - c.first + 1
end


immutable CigarIter
    rs::Reads
    aln::Alignment
end


function Base.length(ci::CigarIter)
    return cigar_len(ci.aln)
end


function Base.start(ci::CigarIter)
    return 1, Int32(ci.aln.leftpos)
end


@inline function Base.next(ci::CigarIter, state::Tuple{Int, Int32})
    i, pos = state
    if i == 1 && length(ci.aln.cigaridx) <= 0
        return (CigarInterval(ci.aln.leftpos, ci.aln.rightpos, OP_MATCH),
                (i + 1, Int32(ci.aln.rightpos + 1)))
    else
        x = ci.rs.cigardata[ci.aln.cigaridx.start + i - 1]
        op = Operation(x & 0x0f)
        len = Int32(x >> 4)
        first = pos
        last = first + len - 1
        return (CigarInterval(pos, pos+len-1, op), (i + 1, Int32(pos+len)))
    end
end


@inline function Base.done(ci::CigarIter, state::Tuple{Int, Int32})
    i, pos = state
    return i > cigar_len(ci.aln)
end


"""
Provide a convenient way to work with iterators.
"""
macro next!(T, x, it, state)
    quote
        if done($(esc(it)), $(esc(state)))
            $(esc(x)) = Nullable{$T}()
        else
            x_, $(esc(state)) = next($(esc(it)), $(esc(state)))
            $(esc(x)) = Nullable{$T}(x_)
        end
    end
end


function is_exon_compatible(op::Operation)
    return op == OP_MATCH || op == OP_SOFT_CLIP ||
           op == OP_INSERT || op == OP_DELETE
end


function is_intron_compatible(op::Operation)
    return op == OP_SKIP || op == OP_SOFT_CLIP
end


macro fragmentlength_next_exonintron!(exons, idx, is_exon, first, last)
    quote
        if $(esc(is_exon))
            if $(esc(idx)) + 1 <= length($(esc(exons)))
                $(esc(first)) = $(esc(exons))[$(esc(idx))].last + 1
                $(esc(last)) = $(esc(exons))[$(esc(idx))+1].first - 1
            else
                $(esc(idx)) += 1
            end
        else
            $(esc(idx)) += 1
            $(esc(first)) = $(esc(exons))[$(esc(idx))].first
            $(esc(last)) = $(esc(exons))[$(esc(idx))].last
        end
        $(esc(is_exon)) = !$(esc(is_exon))
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
        return Nullable{Int}()
    end

    # set a1, a2 as leftmost and rightmost alignments
    a1 = Nullable{Alignment}()
    a2 = Nullable{Alignment}()
    if alnpr.metadata.mate1_idx > 0 && alnpr.metadata.mate2_idx > 0
        mate1 = rs.alignments[alnpr.metadata.mate1_idx]
        mate2 = rs.alignments[alnpr.metadata.mate2_idx]
        if mate1.leftpos <= mate2.leftpos
            a1, a2 = Nullable(mate1), Nullable(mate2)
        else
            a1, a2 = Nullable(mate2), Nullable(mate1)
        end
    elseif alnpr.metadata.mate1_idx > 0
        a1 = Nullable(rs.alignments[alnpr.metadata.mate1_idx])
    else
        a1 = Nullable(rs.alignments[alnpr.metadata.mate2_idx])
    end

    c1_iter = CigarIter(rs, get(a1))
    c1_state = start(c1_iter)
    c1 = Nullable{CigarInterval}()
    @next!(CigarInterval, c1, c1_iter, c1_state)

    exons = t.metadata.exons
    first_exon_idx = searchsortedlast(exons, alnpr)
    e1_idx = first_exon_idx
    e1_isexon = true
    e1_first, e1_last = exons[e1_idx].first, exons[e1_idx].last

    # amount of intronic dna spanned by the fragment
    intronlen = 0

    # skip any leading soft clipping
    if !isnull(c1) && get(c1).op == OP_SOFT_CLIP
        @next!(CigarInterval, c1, c1_iter, c1_state)
    end

    while e1_idx <= length(exons) && !isnull(c1)
        c = get(c1)

        # case 1: e entirely precedes
        if e1_last < c.first
            @fragmentlength_next_exonintron!(exons, e1_idx, e1_isexon,
                                             e1_first, e1_last)

        # case 2: c is contained within e
        elseif c.last >= e1_first && c.last <= e1_last && c.first >= e1_first
            if e1_isexon
                if !is_exon_compatible(c.op)
                    return Nullable{Int}()
                end
            else
                if !is_intron_compatible(c.op)
                    return Nullable{Int}()
                end
                intronlen += e1_last - e1_first + 1
            end
            @next!(CigarInterval, c1, c1_iter, c1_state)

        # case 3: soft clipping partiallly overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            @next!(CigarInterval, c1, c1_iter, c1_state)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e1_last && c.op == OP_MATCH
            if e1_isexon && c.last - e1_last <= max_allowable_encroachment
                c1 = Nullable(CigarInterval(c.first, e1_last, c.op))
            elseif !e1_isexon && e1_last >= c.first &&
                   e1_last - c.first < max_allowable_encroachment
                c1 = Nullable(CigarInterval(e1_last + 1, c.last, c.op))
            else
                return Nullable{Int}()
            end

        # case 5: c precedes and partially overlaps e
        else
            return Nullable{Int}()
        end
    end

    if !isnull(c1)
        return Nullable{Int}()
    end

    # alignment is compatible, but single-ended
    if isnull(a2)
        return Nullable{Int}()
    end

    e2_sup_e1 = false # marks with e2 > e1

    c2_iter = CigarIter(rs, get(a2))
    c2_state = start(c2_iter)
    c2 = Nullable{CigarInterval}()
    @next!(CigarInterval, c2, c2_iter, c2_state)

    e2_idx = first_exon_idx
    e2_isexon = true
    e2_first, e2_last = exons[e2_idx].first, exons[e2_idx].last

    while e2_idx <= length(exons) && !isnull(c2)
        c = get(c2)

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
                    return Nullable{Int}()
                end
            else
                if !is_intron_compatible(c.op)
                    return Nullable{Int}()
                end
            end
            @next!(CigarInterval, c2, c2_iter, c2_state)

        # case 3: soft clipping partially overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            @next!(CigarInterval, c2, c2_iter, c2_state)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e2_last && c.op == OP_MATCH
            if e2_isexon && c.last - e2_last <= max_allowable_encroachment
                c2 = Nullable(CigarInterval(c.first, e2_last, c.op))
            elseif !e2_isexon && e2_last >= c.first &&
                e2_last - c.first < max_allowable_encroachment
                c2 = Nullable(CigarInterval(e2_last + 1, c.last, c.op))
            else
                return Nullable{Int}()
            end

        # case 5: c precedes and partially overlaps e
        else
            return Nullable{Int}()
        end
    end

    # skip any trailing soft clipping
    if !isnull(c2) && get(c2).op == OP_SOFT_CLIP
        @next!(CigarInterval, c2, c2_iter, c2_state)
    end

    if !isnull(c2)
        return Nullable{Int}()
    end

    a1_, a2_ = get(a1), get(a2)
    fraglen = max(a1_.rightpos, a2_.rightpos) -
              min(a1_.leftpos, a2_.leftpos) + 1 - intronlen

    if fraglen > 0
        return Nullable{Int}(fraglen)
    else
        return Nullable{Int}()
    end
end


