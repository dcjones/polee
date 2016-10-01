
# I thought it would be nice to but this in an intervalcollection and intersect
# with transcripts, but 
immutable Alignment
    id::UInt
    hitnum::Int32
    refidx::Int32
    leftpos::Int32
    rightpos::Int32
    # TODO: other alignment info
    #  - alignment number
end


immutable AlignmentPairMetadata
    # index range into the alignments array
    span::UnitRange{Int32}
end
typealias AlignmentPair Interval{AlignmentPairMetadata}


function Base.isless(a::Alignment, b::Alignment)
    return a.id < b.id || (a.id == b.id && a.hitnum < b.hitnum)
end


type Reads
    function Reads(filename)
        prog = Progress(filesize(filename), 0.25, "Reading BAM file ", 60)
        reader = open(BAMReader, filename)
        entry = eltype(reader)()
        readnames = HatTrie()
        alignments = Alignment[]
        # intern sequence names
        seqnames = Dict{String, String}()

        i = 0
        while !isnull(tryread!(reader, entry))
            if (i += 1) % 10000 == 0
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
                                        leftposition(entry), rightposition(entry)))
        end
        finish!(prog)

        @printf("Read %9d reads\nwith %9d alignments\n",
                length(readnames), length(alignments))

        seqnames = [StringField(s) for s in reader.refseqnames]

        # group alignments into alignment pair intervals
        sort!(alignments)
        #alignment_pairs_vec = AlignmentPair[]

        i, j = 1, 1
        prog = Progress(length(alignments), 0.25, "Indexing alignments ", 60)
        alignment_pairs = IntervalCollection{AlignmentPairMetadata}()
        while i <= length(alignments)
            j = i
            update!(prog, i)
            while j + 1 <= length(alignments) &&
                    alignments[i].id == alignments[j+1].id
                j += 1
            end

            seqname = reader.refseqnames[alignments[i].refidx]
            minpos, maxpos = 0, 0
            for k in i:j
                if minpos == 0 || minpos > alignments[k].leftpos
                    minpos = alignments[k].leftpos
                end

                if maxpos == 0 || maxpos < alignments[k].rightpos
                    maxpos = alignments[k].rightpos
                end
            end

            alnpr = AlignmentPair(
                seqname, minpos, maxpos, STRAND_BOTH,
                AlignmentPairMetadata(Int32(i):Int32(j)))
            push!(alignment_pairs, alnpr)

            i = j + 1
        end
        finish!(prog)

        #alignment_pairs = IntervalCollection(alignment_pairs_vec, true)
        @show length(alignment_pairs)

        return new()
    end
end

