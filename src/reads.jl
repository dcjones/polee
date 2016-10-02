
# I thought it would be nice to but this in an intervalcollection and intersect
# with transcripts, but 
immutable Alignment
    id::UInt
    hitnum::Int32
    refidx::Int32
    leftpos::Int32
    rightpos::Int32
    flag::UInt16
    # TODO: other alignment info
end


immutable AlignmentPairMetadata
    mate1_idx::UInt32
    mate2_idx::UInt32
end
typealias AlignmentPair Interval{AlignmentPairMetadata}


function Base.isless(a::Alignment, b::Alignment)
    return a.id < b.id || (a.id == b.id && a.hitnum < b.hitnum)
end


type Reads
    alignments::Vector{Alignment}
    alignment_pairs::IntervalCollection{AlignmentPairMetadata}
end


function Reads(filename::String)
    prog = Progress(filesize(filename), 0.25, "Reading BAM file ", 60)
    reader = open(BAMReader, filename)
    entry = eltype(reader)()
    readnames = HatTrie()
    alignments = Alignment[]
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
                                    flag(entry)))
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

    return Reads(alignments, alignment_pairs)
end
