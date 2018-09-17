
# TODO: Thin wrappers for tensorflow models.

function estimate_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input)
    elseif input.feature == :gene
        return estimate_gene_expression(input)
    elseif input.feature == :splicing
        return estimate_splicing_proportions(input)
    else
        error("Expression estimates for $(input.feature) not supported")
    end
end


function estimate_transcript_expression(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    x0_log = log.(input.loaded_samples.x0_values)
    polee_py[:estimate_transcript_expression](
        input.loaded_samples.init_feed_dict, num_samples, n,
        input.loaded_samples.variables, x0_log)
end


function estimate_mixture(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_mixture(input)
    else
        error("Expression estimates for $(input.feature) not supported")
    end
end


function estimate_transcript_mixture(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    x0_log = log.(input.loaded_samples.x0_values)

    # TODO: how does this get passed in?
    num_components = 4

    polee_py[:estimate_transcript_mixture](
        input.loaded_samples.init_feed_dict, num_samples, n,
        input.loaded_samples.variables, x0_log, num_components)
end


const POLEE_MODELS = Dict(
    "expression" => estimate_expression,
    "mixture"    => estimate_mixture
)