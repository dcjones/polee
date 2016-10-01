

immutable Exon <: Intervals.AbstractInterval{Int64}
    first::Int64
    last::Int64
end


type TranscriptMetadata
    name::StringField
    exons::Vector{Exon}

    function TranscriptMetadata(name)
        return new(name, Exon[])
    end
end


typealias Transcript Interval{TranscriptMetadata}


function Base.push!(t::Transcript, e::Exon)
    push!(t.metadata.exons, e)
    t.first = min(t.first, e.first)
    t.last = min(t.first, e.last)
    return e
end


type Transcripts
    transcripts::IntervalCollection{TranscriptMetadata}
    transcripts_by_name::Dict{StringField, Transcript}

    function Transcripts(filename::String)
        prog = Progress(filesize(filename), 0.25, "Reading GFF3 file ", 60)
        reader = open(GFF3Reader, filename)
        entry = eltype(reader)()
        transcripts_by_name = Dict{StringField, Transcript}()

        i = 0
        while !isnull(tryread!(reader, entry))
            if (i += 1) % 10000 == 0
                update!(prog, position(reader.state.stream.source))
            end
            if entry.metadata.kind == "exon"
                if !haskey(entry.metadata.attributes, "Parent")
                    error("Exon has no parent")
                end

                parent_name = entry.metadata.attributes["Parent"]
                # TODO: support ensembl style "transcript:id" syntax?
                if !haskey(transcripts_by_name, parent_name)
                    parent_name = copy(parent_name)
                    transcripts_by_name[parent_name] =
                        Transcript(entry.seqname, entry.first, entry.last,
                                   entry.strand, TranscriptMetadata(parent_name))
                end
                push!(transcripts_by_name[parent_name], Exon(entry.first, entry.last))
            end
        end
        finish!(prog)
        println("Read ", length(transcripts_by_name), " transcripts")
        is = collect(values(transcripts_by_name))
        transcripts = IntervalCollection(is, true)

        return new(transcripts, transcripts_by_name)
    end
end


function genomic_to_transcriptomic(geneset::Transcripts, position::Int)

end

# TODO: same but with alignments
#function genomic_to_transcriptomic(geneset::GeneSet, alignment)
#end

