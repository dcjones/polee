
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
    while !isnull(tryread!(reader, entry))
        seqname = FASTA.identifier(entry)
        if length(entry.sequence) > 100000
            ProgressMeter.update!(prog, position(reader.state.stream.source))
        end

        if haskey(ts.trees, seqname)
            entryseq = FASTA.sequence(entry)
            for t in ts.trees[seqname]
                seq = t.metadata.seq
                for exon in t.metadata.exons
                    if exon.last <= length(entryseq)
                        append!(seq, entryseq[exon.first:exon.last])
                    end
                end
                if t.strand == STRAND_NEG
                    reverse_complement!(seq)
                end
            end
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
            seq = t.metadata.seq
            for exon in t.metadata.exons
                if exon.last <= length(refseq)
                    append!(seq, DNASequence(refseq[exon.first:exon.last]))
                end
            end
            if t.strand == STRAND_NEG
                reverse_complement!(seq)
            end
        end
    end
end

