
function read_transcript_sequences!(ts, filename)
    if endswith(filename, ".2bit")
        read_transcript_sequences_from_twobit!(ts, filename)
    else
        read_transcript_sequences_from_fasta!(ts, filename)
    end
end


function read_transcript_sequences_from_fasta!(ts, filename)
    prog = Progress(filesize(filename), 0.25, "Reading sequences ", 60)
    reader = open(FASTA.Reader, filename)
    entry = eltype(reader)()

    i = 0
    seen_seqs = Set{String}()
    while tryread!(reader, entry)
        seqname = FASTA.identifier(entry)
        if length(entry.sequence) > 100000
            ProgressMeter.update!(prog, position(reader.state.stream))
        end

        if haskey(ts.trees, seqname)
            push!(seen_seqs, seqname)
            entryseq = FASTA.sequence(entry)
            for t in ts.trees[seqname]
                seq = LongDNASeq()
                for exon in t.metadata.exons
                    if exon.last <= length(entryseq)
                        append!(seq, entryseq[exon.first:exon.last])
                    end
                end
                if t.strand == STRAND_NEG
                    reverse_complement!(seq)
                end

                t.metadata.seq = convert(Vector{DNA}, seq)
            end
        end
    end

    for seqname in keys(ts.trees)
        if seqname ∉ seen_seqs
            @warn string("FASTA file has sequence for ", seqname)
        end
    end

    finish!(prog)
end


function read_transcript_sequences_from_twobit!(ts, filename)
    reader = open(TwoBit.Reader, filename)
    prog = Progress(length(ts.trees), 0.25, "Reading sequences ", 60)

    for (i, (name, tree)) in enumerate(ts.trees)
        ProgressMeter.update!(prog, i)
        local refseq
        try
            refseq = reader[name].seq
        catch
            continue
        end

        for t in tree
            seq = LongDNASeq()
            for exon in t.metadata.exons
                if exon.last <= length(refseq)
                    append!(seq, LongDNASeq(entryseq[exon.first:exon.last]))
                end
            end
            if t.strand == STRAND_NEG
                reverse_complement!(seq)
            end

            t.metadata.seq = convert(Vector{DNA}, seq)
        end
    end
end

