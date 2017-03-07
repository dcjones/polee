

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
function _leftposition(rec::BAMRecord)
    pos = leftposition(rec)
    offset = Align.seqname_length(rec)
    for i in offset+1:4:offset+Align.n_cigar_op(rec)*4
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


function _alignment_length(rec::BAMRecord)
    offset = Align.seqname_length(rec)
    length::Int = 0
    for i in offset+1:4:offset+Align.n_cigar_op(rec)*4
        x = unsafe_load(Ptr{UInt32}(pointer(rec.data, i)))
        op = Operation(x & 0x0f)
        if ismatchop(op) || isdeleteop(op) || op == OP_SOFT_CLIP
            length += x >> 4
        end
    end
    return length
end


function _rightposition(rec::BAMRecord)
    return Int32(_leftposition(rec) + _alignment_length(rec) - 1)
end


immutable AlignmentPairMetadata
    mate1_idx::UInt32
    mate2_idx::UInt32
end
typealias AlignmentPair Interval{AlignmentPairMetadata}


function Base.isless(a::Alignment, b::Alignment)
    return a.id < b.id ||
        (a.id == b.id && (a.flag & SAM_FLAG_READ2) < (b.flag & SAM_FLAG_READ2))
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


function Reads(filename::String)
    prog_step = 1000
    if filename == "-"
        prog = Progress(0, 0.25, "Reading BAM file ", 60)
        reader = BAMReader(STDIN)
        from_file = false
    else
        prog = Progress(filesize(filename), 0.25, "Reading BAM file ", 60)
        reader = open(BAMReader, filename)
        from_file = true
    end
    entry = eltype(reader)()
    readnames = HatTrie()
    alignments = Alignment[]
    cigardata = UInt32[]
    # intern sequence names
    seqnames = Dict{String, String}()

    i = 0
    while !isnull(tryread!(reader, entry))
        if from_file && (i += 1) % prog_step == 0
            update!(prog, position(reader.stream.io))
        end

        if !ismapped(entry)
            continue
        end

        # copy cigar data over if there are any non-match operations
        cigarptr = Ptr{UInt32}(pointer(
                entry.data, 1 + Align.seqname_length(entry)))
        cigarlen = Align.n_cigar_op(entry)

        N = UInt32(length(cigardata))
        cigaridx = N+1:N
        if cigarlen > 1 || cigar_from_ptr(cigarptr, 1)[1] != OP_MATCH
            cigaridx = N+1:N+1+cigarlen-1
            resize!(cigardata, N + cigarlen)
            unsafe_copy!(pointer(cigardata, N + 1),
                         cigarptr, cigarlen)
        end

        id = get!(readnames, seqname(entry), length(readnames) + 1)
        push!(alignments, Alignment(id, entry.refid + 1,
                                    _leftposition(entry), _rightposition(entry),
                                    flag(entry), cigaridx))
    end
    finish!(prog)

    @printf("Read %9d reads\nwith %9d alignments\n",
            length(readnames), length(alignments))

    seqnames = [StringField(s) for s in reader.refseqnames]

    # group alignments into alignment pair intervals
    sort!(alignments)

    i, j = 1, 1
    prog = Progress(1 + div(length(alignments), prog_step), 0.25,
                    "Indexing alignments ", 60)
    alignment_pairs = IntervalCollection{AlignmentPairMetadata}()
    while i <= length(alignments)
        if i % prog_step == 0
            next!(prog)
        end

        j1 = i
        while j1 + 1 <= length(alignments) &&
              alignments[i].id == alignments[j1+1].id &&
              (alignments[j1+1].flag & SAM_FLAG_READ2) == 0
            j1 += 1
        end

        j2 = j1
        while j2 + 1 <= length(alignments) &&
              alignments[i].id == alignments[j2+1].id
            j2 += 1
        end

        # now i:j1 are mate1 alignments, and j1+1:j2 are mate2 alignments

        seqname = seqnames[alignments[i].refidx]

        # examine every potential mate1, mate2 pair
        for k1 in i:j1
            m1 = alignments[k1]
            for k2 in j1+1:j2
                m2 = alignments[k2]

                if m1.refidx != m2.refidx ||
                   (m1.flag & SAM_FLAG_REVERSE) == (m2.flag & SAM_FLAG_REVERSE)
                    continue
                end

                minpos = min(m1.leftpos, m2.leftpos)
                maxpos = max(m1.rightpos, m2.rightpos)

                if maxpos - minpos > MAX_PAIR_DISTANCE
                    continue
                end

                strand = m1.flag & SAM_FLAG_REVERSE != 0 ? STRAND_NEG : STRAND_POS

                alnpr = AlignmentPair(
                    seqname, minpos, maxpos, strand,
                    AlignmentPairMetadata(k1, k2))
                push!(alignment_pairs, alnpr)
            end
        end

        # handle single-end reads
        if isempty(j1+1:j2)
            for k in i:j1
                m = alignments[k]
                if m.flag & SAM_FLAG_READ1 != 0
                    continue
                end

                # TODO
            end
        end

        i = j2 + 1
    end
    finish!(prog)

    return Reads(alignments, alignment_pairs, cigardata)
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


function Base.next(ci::CigarIter, state::Tuple{Int, Int32})
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


function Base.done(ci::CigarIter, state::Tuple{Int, Int32})
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

    e1_iter = ExonIntronIter(t)
    e1_state = start(e1_iter)
    e1 = Nullable{ExonIntron}()
    @next!(ExonIntron, e1, e1_iter, e1_state)

    intronlen = 0

    # skip any leading soft clipping
    if !isnull(c1) && get(c1).op == OP_SOFT_CLIP
        @next!(CigarInterval, c1, c1_iter, c1_state)
    end

    while !isnull(e1) && !isnull(c1)
        c = get(c1)
        e = get(e1)

        # case 1: e entirely precedes
        if e.last < c.first
            @next!(ExonIntron, e1, e1_iter, e1_state)

        # case 2: c is contained within e
        elseif c.last >= e.first && c.last <= e.last && c.first >= e.first
            if e.isexon
                if !is_exon_compatible(c.op)
                    return Nullable{Int}()
                end
            else
                if !is_intron_compatible(c.op)
                    return Nullable{Int}()
                end
                intronlen += length(e)
                @next!(ExonIntron, e1, e1_iter, e1_state)
            end
            @next!(CigarInterval, c1, c1_iter, c1_state)

        # case 3: soft clipping partiallly overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            @next!(CigarInterval, c1, c1_iter, c1_state)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e.last && c.op == OP_MATCH
            if e.isexon && c.last - e.last <= max_allowable_encroachment
                c1 = Nullable(CigarInterval(c.first, e.last, c.op))
            elseif !e.isexon && e.last >= c.first &&
                   e.last - c.first < max_allowable_encroachment
                c1 = Nullable(CigarInterval(e.last + 1, c.last, c.op))
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

    e2_iter = ExonIntronIter(t)
    e2_state = start(e2_iter)
    e2 = Nullable{ExonIntron}()
    @next!(ExonIntron, e2, e2_iter, e2_state)


    while !isnull(e2) && !isnull(c2)
        c = get(c2)
        e = get(e2)

        # case 1: e entirely precedes c
        if e.last < c.first
            if !e.isexon && e2_sup_e1
                intronlen += length(e)
            end

            if !isnull(e1) && e == get(e1)
                e2_sup_e1 = true
            end

            @next!(ExonIntron, e2, e2_iter, e2_state)

        # case 2: c is contained within e
        elseif c.last >= e.first && c.last <= e.last && c.first >= e.first
            if e.isexon
                if !is_exon_compatible(c.op)
                    return Nullable{Int}()
                end
            else
                if !is_intron_compatible(c.op)
                    return Nullable{Int}()
                end
                # TODO: why not increment e2 here?
            end
            @next!(CigarInterval, c2, c2_iter, c2_state)

        # case 3: soft clipping partially overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            @next!(CigarInterval, c2, c2_iter, c2_state)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e.last && c.op == OP_MATCH
            if e.isexon && c.last - e.last <= max_allowable_encroachment
                c2 = Nullable(CigarInterval(c.first, e.last, c.op))
            elseif !e.isexon && e.last >= c.first &&
                e.last - c.first < max_allowable_encroachment
                c2 = Nullable(CigarInterval(e.last + 1, c.last, c.op))
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


