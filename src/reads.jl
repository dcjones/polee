
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
            m1 = m2 = i
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


immutable CigarIter
    rs::Reads
    aln::Alignment
end


function Base.start(ci::CigarIter)
    return 1
end

function Base.next(ci::CigarIter, i)
    if i == 1 && length(ci.aln.cigaridx) <= 0
        return ((OP_MATCH, ci.aln.rightpos - ci.aln.leftpos + 1), i + 1)
    else
        x = ci.rs.cigardata[ci.aln.cigaridx.start + i - 1][]
        return ((Operation(x & 0x0f), Int32(x >> 4)), i + 1)
    end
end

function Base.done(ci::CigarIter, i)
    return i > cigar_len(ci.aln)
end

