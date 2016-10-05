
immutable Alignment
    id::UInt32
    hitnum::Int32
    refidx::Int32
    leftpos::Int32
    rightpos::Int32
    flag::UInt16
    # indexes into Reads.cigardata encoding alignment
    cigaridx::UnitRange{UInt32}
end


immutable AlignmentPairMetadata
    mate1_idx::UInt32
    mate2_idx::UInt32
end
typealias AlignmentPair Interval{AlignmentPairMetadata}


function Base.isless(a::Alignment, b::Alignment)
    return a.id < b.id || (a.id == b.id && a.hitnum < b.hitnum)
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
    prog = Progress(filesize(filename), 0.25, "Reading BAM file ", 60)
    reader = open(BAMReader, filename)
    entry = eltype(reader)()
    readnames = HatTrie()
    alignments = Alignment[]
    cigardata = UInt32[]
    # intern sequence names
    seqnames = Dict{String, String}()

    i = 0
    while !isnull(tryread!(reader, entry))
        if (i += 1) % 1000 == 0
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

        hitnum = 0
        aux = Align.optional_fields(entry)
        try
            hitnum = aux["HI"]
        catch ex
            if !isa(ex, KeyError)
                rethrow()
            end
        end

        id = get!(readnames, seqname(entry), length(readnames) + 1)
        # TODO: I think I need to compute rightposition *with* soft clipping
        push!(alignments, Alignment(id, hitnum, entry.refid + 1,
                                    leftposition(entry), rightposition(entry),
                                    flag(entry), cigaridx))
    end
    finish!(prog)

    @printf("Read %9d reads\nwith %9d alignments\n",
            length(readnames), length(alignments))

    seqnames = [StringField(s) for s in reader.refseqnames]

    # group alignments into alignment pair intervals
    sort!(alignments)
    i, j = 1, 1
    prog = Progress(length(alignments), 0.25, "Indexing alignments ", 60)
    alignment_pairs = IntervalCollection{AlignmentPairMetadata}()
    while i <= length(alignments)
        if i % 1000 == 0
            update!(prog, i)
        end

        j = i
        while j + 1 <= length(alignments) &&
                alignments[i].id == alignments[j+1].id &&
                alignments[i].hitnum == alignments[j+1].hitnum
            j += 1
        end

        if j > i + 1
            error("Alignment with more than two mates found.")
        end

        seqname = reader.refseqnames[alignments[i].refidx]
        minpos = min(alignments[i].leftpos, alignments[j].leftpos)
        maxpos = max(alignments[i].rightpos, alignments[j].rightpos)

        m1 = m2 = 0
        if i == j
            if alignments[i].flag & SAM_FLAG_READ2
                m2 = i
            else
                m1 = i
            end
        elseif alignments[i].flag & SAM_FLAG_READ1 != 0 &&
               alignments[j].flag & SAM_FLAG_READ2 != 0
            m1, m2 = i, j
        elseif alignments[i].flag & SAM_FLAG_READ2 != 0 &&
               alignments[j].flag & SAM_FLAG_READ1 != 0
            m1, m2 = j, i
        else
            error("Alignment pair has incorrect flags set.")
        end

        alnpr = AlignmentPair(
            seqname, minpos, maxpos, STRAND_BOTH,
            AlignmentPairMetadata(m1, m2))
        push!(alignment_pairs, alnpr)

        i = j + 1
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


function Base.start(ci::CigarIter)
    return 1, ci.aln.leftpos
end

function Base.next(ci::CigarIter, state::Tuple{Int, Int32})
    i, pos = state
    if i == 1 && length(ci.aln.cigaridx) <= 0
        return (CigarInterval(ci.aln.leftpos, ci.aln.rightpos, OP_MATCH),
                (i + 1, ci.aln.rightpos + 1))
    else
        x = ci.rs.cigardata[ci.aln.cigaridx.start + i - 1]
        op = Operation(x & 0x0f)
        len = Int32(x >> 4)
        first = pos
        last = first + len - 1
        return (CigarInterval(pos, pos+len-1, op), (i + 1, pos+len))
    end
end

function Base.done(ci::CigarIter, state::Tuple{Int, Int32})
    i, pos = state
    return i > cigar_len(ci.aln)
end


"""
Provide a slightly different iterator interface.
"""
type StoredIterator{T,S,V}
    it::T
    state::S
    value::Nullable{V}

    function StoredIterator(it)
        state = start(it)
        if !done(it, state)
            value, state = next(it, state)
            return new(it, state, Nullable{V}(value))
        else
            return new(it, state, Nullable{V}())
        end
    end
end


function Base.get(si::StoredIterator)
    return get(si.value)
end


function Base.isnull(si::StoredIterator)
    return isnull(si.value)
end


function next!{T,S,V}(si::StoredIterator{T,S,V})
    if done(si.it, si.state)
        si.value = Nullable{V}()
    else
        value, si.state = next(si.it, si.state)
        si.value = Nullable{V}(value)
    end
    return si.value
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

    c1 = StoredIterator{CigarIter, Tuple{Int,Int32}, CigarInterval}(
                CigarIter(rs, get(a1)))
    e1 = StoredIterator{ExonIntronIter, Tuple{Int, Bool}, ExonIntron}(
                ExonIntronIter(t))
    intronlen = 0

    # skip any leading soft clipping
    if !isnull(c1) && get(c1).op == OP_SOFT_CLIP
        next!(c1)
    end

    while !isnull(e1) && !isnull(c1)
        c = get(c1)
        e = get(e1)

        # case 1: e entirely precedes
        if e.last < c.first
            next!(e1)

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
                next!(e1)
            end
            next!(c1)

        # case 3: soft clipping partiallly overlapping an exon or intron
        elseif c.op == OP_SOFT_CLIP
            next!(c1)

        # case 4: match op overhangs into an intron a little
        elseif c.last > e.last && c.op == OP_MATCH
            if e.isexon && c.last - e.last <= max_allowable_encroachment
                c1.value = Nullable(CigarInterval(c.first, e.last, c.op))
            elseif !e.isexon && e.last >= c.first &&
                   e.last - c.first < max_allowable_encroachment
                c1.value = Nullable(CigarInterval(e.last + 1, c.last, c.op))
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
    if !isnull(a2)
        return Nullable{0}
    end

    # TODO: now process a2

    return Nullable{Int}()
end


