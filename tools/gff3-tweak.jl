#!/usr/bin/env julia

# This program modifies gene annotations to make them "safer".
#
# RNA-Seq quantification that is focused on transcript expression (like Polee)
# can break when the annotated transcripts are not correct. This is particularly
# a problem when the true expressed transcript is not annotated. This can
# manifest as incorrect predictions of alternative splicing.
#
# Transcript start and terminitation sites that differ from annotations is one
# of the main ways this occurs. This problem tweaks the gff file by extending
# the initial and terminal exon of spliced transcripts to match the longest
# initial/terminal compatible exon. This can cause the quantification to miss
# start and end site switiching, but I think this is a worthwhile tradeoff.

polee_dir = joinpath(dirname(@__FILE__), "..")

import Pkg
Pkg.activate(polee_dir)

import Polee
using GenomicFeatures


function find_duplicate_transcripts(ts)
    excluded_transcripts = Set{String}()
    for t in ts
        for u in eachoverlap(ts, t)
            if t.metadata.id > u.metadata.id &&
                    t.strand == u.strand &&
                    t.metadata.exons == u.metadata.exons
                push!(excluded_transcripts, t.metadata.name)
            end
        end
    end
    return excluded_transcripts
end


function main()
    filename = ARGS[1]
    ts, ts_metadata = Polee.Transcripts(filename)

    gene_initial_exons = Dict{String, Set{Polee.Exon}}()
    gene_terminal_exons = Dict{String, Set{Polee.Exon}}()

    for t in ts
        if length(t.metadata.exons) <= 1
            continue
        end

        if !haskey(ts_metadata.gene_id, t.metadata.name)
            continue
        end

        gene_id = ts_metadata.gene_id[t.metadata.name]

        if !haskey(gene_initial_exons, gene_id)
            gene_initial_exons[gene_id] = Set{Polee.Exon}()
            gene_terminal_exons[gene_id] = Set{Polee.Exon}()
        end

        push!(gene_initial_exons[gene_id], t.metadata.exons[1])
        push!(gene_terminal_exons[gene_id], t.metadata.exons[end])
    end

    # (gene_id, first, last) -> (new_first, new_last)
    first_exon_adjustments = Dict{Tuple{String, Int, Int}, Polee.Exon}()
    for (gene_id, exons) in gene_initial_exons
        for exon_a in exons
            min_first = exon_a.first
            for exon_b in exons
                if exon_a.last == exon_b.last && exon_b.first < min_first
                    min_first = exon_b.first
                end
            end

            if min_first != exon_a.first
                first_exon_adjustments[(gene_id, exon_a.first, exon_a.last)] =
                    Polee.Exon(min_first, exon_a.last)
            end
        end
    end

    last_exon_adjustments = Dict{Tuple{String, Int, Int}, Polee.Exon}()
    for (gene_id, exons) in gene_terminal_exons
        for exon_a in exons
            max_last = exon_a.last
            for exon_b in exons
                if exon_a.first == exon_b.first && exon_b.last > max_last
                    max_last = exon_b.last
                end
            end

            if max_last != exon_a.last
                last_exon_adjustments[(gene_id, exon_a.first, exon_a.last)] =
                    Polee.Exon(exon_a.first, max_last)
            end
        end
    end

    # Find transcripts that are made redundant by exon adjustments
    ts2 = Polee.Transcripts()
    for t in ts
        if length(t.metadata.exons) <= 1 || !haskey(ts_metadata.gene_id, t.metadata.name)
            push!(ts2, t)
            continue
        end

        gene_id = ts_metadata.gene_id[t.metadata.name]
        first_exon, last_exon = t.metadata.exons[1], t.metadata.exons[end]
        if haskey(first_exon_adjustments, (gene_id, first_exon.first, first_exon.last))
            first_exon = first_exon_adjustments[(gene_id, first_exon.first, first_exon.last)]
        end

        if haskey(last_exon_adjustments, (gene_id, last_exon.first, last_exon.last))
            last_exon = last_exon_adjustments[(gene_id, last_exon.first, last_exon.last)]
        end

        t2 = Polee.Transcript(
            t.seqname, first_exon.first, last_exon.last, t.strand,
            Polee.TranscriptMetadata(
                t.metadata.name, t.metadata.id,
                copy(t.metadata.exons), t.metadata.seq))

        @assert t2.metadata.exons[1].first >= first_exon.first
        @assert t2.metadata.exons[1].last == first_exon.last

        @assert t2.metadata.exons[end].first == last_exon.first
        @assert t2.metadata.exons[end].last <= last_exon.last

        t2.metadata.exons[1] = first_exon
        t2.metadata.exons[end] = last_exon
        push!(ts2, t2)
    end

    excluded_transcripts = find_duplicate_transcripts(ts2)

    transcript_map = Dict{String, Polee.Transcript}()
    for t in ts
        transcript_map[t.metadata.name] = t
    end

    # do the adjustment
    parent_pat = r"Parent=([^;]*);"
    id_pat = r"ID=([^;]*);"

    for line in eachline(filename)
        if line[1] == '#'
            println(line)
            continue
        end
        row = split(line, '\t')

        parent_mat = match(parent_pat, row[9])
        id_mat = match(id_pat, row[9])

        if parent_mat !== nothing && in(parent_mat.captures[1], excluded_transcripts)
            continue
        end

        if id_mat !== nothing && id_mat.captures[1] ∈ excluded_transcripts
            continue
        end

        if row[3] == "exon"
            first = parse(Int, row[4])
            last = parse(Int, row[5])
            gene_id = ts_metadata.gene_id[parent_mat.captures[1]]
            curr_exon_tup = (gene_id, first, last)
            curr_exon = Polee.Exon(first, last)
            new_exon = curr_exon

            transcript_exons = transcript_map[parent_mat.captures[1]].metadata.exons
            if transcript_exons[1] == curr_exon &&
                    haskey(first_exon_adjustments, curr_exon_tup)
                new_exon = first_exon_adjustments[curr_exon_tup]
            elseif transcript_exons[end] == curr_exon &&
                    haskey(last_exon_adjustments, curr_exon_tup)
                new_exon = last_exon_adjustments[curr_exon_tup]
            end

            if curr_exon != new_exon
                println(
                    row[1], "\t",
                    row[2], "\t",
                    row[3], "\t",
                    new_exon.first, "\t",
                    new_exon.last, "\t",
                    row[6], "\t",
                    row[7], "\t",
                    row[8], "\t",
                    row[9])
            else
                println(line)
            end
        else
            println(line)
        end
    end
end

main()


