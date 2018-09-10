

# Don't consider read pairs more than this far apart
const MAX_PAIR_DISTANCE = 500000

struct Alignment
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


struct AlignmentPairMetadata
    mate1_idx::UInt32
    mate2_idx::UInt32
end
const AlignmentPair = Interval{AlignmentPairMetadata}


function group_alignments_isless(a::Alignment, b::Alignment)
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


struct Reads
    alignments::Vector{Alignment}
    alignment_pairs::IntervalCollection{AlignmentPairMetadata}
    cigardata::Vector{UInt32}
    num_reads::Int
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
        prog = Progress(0, 0.25, "Reading BAM file ", 60)
        from_file = false
    else
        reader = open(BAM.Reader, filename)
        prog = Progress(filesize(filename), 0.25, "Reading BAM file ", 60)
        from_file = true
    end
    return Reads(reader, prog, from_file, excluded_seqs)
end


function Reads(input::IO, excluded_seqs::Set{String})
    reader = BAM.Reader(input)
    prog = Progress(0, 0.25, "Reading BAM file ", 60)
    from_file = false
    return Reads(reader, prog, from_file, excluded_seqs)
end



function Reads(reader::BAM.Reader, prog::Progress, from_file::Bool,
               excluded_seqs::Set{String})
    prog_step = 1000
    entry = eltype(reader)()
    readnames = Dict{UInt64, UInt32}()
    alignments = Alignment[]
    cigardata = UInt32[]

    # don't bother with reads from sequences with no transcripts
    excluded_refidxs = BitSet()
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
        cigaridx = UInt32(N+1):UInt32(N)
        if cigarlen > 1 || cigar_from_ptr(cigarptr, 1)[1] != OP_MATCH
            # if the last alignment had the same cigar, just use that

            # check if we have the same cigar string as the last alignment
            prev_identical = false
            if length(cigardata) >= cigarlen
                c = ccall(:memcmp, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                          pointer(cigardata, N-cigarlen+1),
                          cigarptr, 4*cigarlen)
                prev_identical = c == 0
            end

            if prev_identical
                cigaridx = UInt32(N-cigarlen+1):UInt32(N)
            else
                cigaridx = UInt32(N+1):UInt32(N+1+cigarlen-1)
                resize!(cigardata, N + cigarlen)
                unsafe_copyto!(
                    pointer(cigardata, N + 1), cigarptr, cigarlen)
            end
        end

        id = get!(readnames, hash(BioAlignments.seqname(entry)), length(readnames) + 1)
        lp = _leftposition(entry)
        rp = _rightposition(entry)
        flg = BAM.flag(entry) & USED_BAM_FLAGS

        # skip if the last alignment was identical
        if !isempty(alignments)
            last_aln = alignments[end]
            if last_aln.id       == id &&
               last_aln.refidx   == entry.refid + 1 &&
               last_aln.leftpos  == lp &&
               last_aln.rightpos == rp &&
               last_aln.flag     == flg &&
               last_aln.cigaridx == cigaridx
                continue
            end
        end

        push!(alignments, Alignment(id, entry.refid + 1, lp, rp, flg, cigaridx))
    end
    finish!(prog)

    @printf("Read %9d reads\nwith %9d alignments\n",
            length(readnames), length(alignments))

    # group alignments into alignment pair intervals
    sort!(alignments, lt=group_alignments_isless)

    # first partition alignments by reference sequence
    blocks = UnitRange{Int}[]
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

    trees = Array{GenomicFeatures.ICTree{AlignmentPairMetadata}}(undef, length(blocks))
    alignment_pairs = make_interval_collection(alignments, reader.refseqnames, blocks, trees, cigardata)

    return Reads(alignments, alignment_pairs, cigardata, length(readnames))
end


function make_interval_collection(alignments, seqnames, blocks, trees, cigardata)
    Threads.@threads for blockidx in 1:length(blocks)
        block = blocks[blockidx]
        seqname = seqnames[alignments[block.start].refidx]
        alnprs = AlignmentPair[]

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
                    push!(alnprs, alnpr)
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
                    push!(alnprs, alnpr)
                    alncnt += 1
                end
            end

            i = j2 + 1
        end

        sort!(alnprs, lt=IntervalTrees.basic_isless)
        tree = GenomicFeatures.ICTree{AlignmentPairMetadata}(alnprs)
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


"""
Sample a subset of rs without replacement, and treating reads that start at the
same position as equivalent.
"""
function subsample_reads(rs::Reads, n::Int)
    # resevoir sampling to select n alignments with unique positions. Unique
    # positions are used to avoid overfitting models to any one highly
    # expressed transcript (e.g. rRNA)
    reservoir = zeros(Int, n)
    next_reservoir_idx = 1
    last_first = 0
    i = 1
    for alnpr in rs.alignment_pairs
        if alnpr.first != last_first
            if next_reservoir_idx <= n
                reservoir[next_reservoir_idx] = i
                next_reservoir_idx += 1
            else
                j = rand(1:i)
                if j <= n
                    reservoir[j] = i
                end
            end
            last_first = alnpr.first
            i += 1
        end
    end

    # build subset of reads
    reservoir_set = Set{Int}(
        next_reservoir_idx <= n ? reservoir[1:next_reservoir_idx-1] : reservoir)

    read_idxs_subset = Set{UInt32}()
    last_first = 0
    i = 1
    for alnpr in rs.alignment_pairs
        if alnpr.first != last_first
            id = rs.alignments[alnpr.metadata.mate1_idx].id
            if i ∈ reservoir_set
                push!(read_idxs_subset, rs.alignments[alnpr.metadata.mate1_idx].id)
            end
            last_first = alnpr.first
            i += 1
        end
    end

    # build collection containing every alignment of reads in the subset
    alignment_pairs_subset = IntervalCollection{AlignmentPairMetadata}()
    for alnpr in rs.alignment_pairs
        id = rs.alignments[alnpr.metadata.mate1_idx].id
        if id ∈ read_idxs_subset
            push!(alignment_pairs_subset, alnpr)
        end
    end

    return Reads(rs.alignments, alignment_pairs_subset, rs.cigardata, length(read_idxs_subset))
end


function cigar_len(aln::Alignment)
    return max(length(aln.cigaridx), 1)
end


struct CigarInterval
    first::Int
    last::Int
    op::Operation
end


function Base.length(c::CigarInterval)
    return c.last - c.first + 1
end


struct CigarIter
    rs::Reads
    aln::Alignment
end


function Base.length(ci::CigarIter)
    return cigar_len(ci.aln)
end


function Base.iterate(ci::CigarIter)
    return iterate(ci, (1, Int32(ci.aln.leftpos)))
end


@inline function Base.iterate(ci::CigarIter, state::Tuple{Int, Int32})
    i, pos = state

    if i > cigar_len(ci.aln)
        return nothing
    end

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

