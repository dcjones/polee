#!/usr/bin/env julia

# Dump a table with a bunch of transcript annotation metadata,
# which is useful for inspecting regression results for interesting
# or concerning patterns.

polee_dir = joinpath(dirname(@__FILE__), "..")

import Pkg
Pkg.activate(polee_dir)

import Polee

function main()
    ts, ts_metadata = Polee.Transcripts(ARGS[1])


    open(ARGS[2], "w") do output
        println(
            output,
            "transcript_id,",
            "seqname,",
            "length,",
            "kind,",
            "biotype,",
            "gene_id,",
            "gene_name,",
            "gene_biotype,",
            "gene_description")
        for t in ts
            tid = t.metadata.name
            gid = get(ts_metadata.gene_id, t.metadata.name, "")
            println(
                output,
                tid, ",",
                t.seqname, ",",
                Polee.exonic_length(t), ",",
                get(ts_metadata.transcript_kind, tid, ""), ",",
                get(ts_metadata.transcript_biotype, tid, ""), ",",
                gid, ",",
                get(ts_metadata.gene_name, gid, ""), ",",
                get(ts_metadata.gene_biotype, gid, ""), ",",
                "\"", get(ts_metadata.gene_description, gid, ""), "\"")
        end
    end
end


main()

