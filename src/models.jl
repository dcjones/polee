
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
    num_pca_components = 4
    num_mix_components = 4

    component_probs = polee_py[:estimate_transcript_mixture](
        input.loaded_samples.init_feed_dict, num_samples, n,
        input.loaded_samples.variables, x0_log,
        num_mix_components, num_pca_components)

    write_component_membership_html(input, component_probs)
end


function write_component_membership_html(input::ModelInput, component_probs)
    num_samples, num_components = size(component_probs)
    open("component_membership.html", "w") do output
        println(output,
            """
            <!html>
            <head><title>component assignment probabilities</title>
            <style>
            td {
                border: 1px solid;
            }
            </style>
            </head>
            <body>
            <table>
            """)

        println(output, "<tr>")
        println(output, "<td></td>")
        for i in 1:num_components
            println(output, "<td>", i, "</td>")
        end
        println(output, "</tr>")

        for j in 1:num_samples
            println(output, "<tr>")
            println(output, "<td>", input.loaded_samples.sample_names[j], "</td>")
            for i in 1:num_components
                if component_probs[j, i] > 1e-10
                    println(output, "<td>", component_probs[j, i], "</td>")
                else
                    println(output, "<td></td>")
                end
            end

            println(output, "</tr>")
        end

        println(output,
            """
            </table></body>
            """)
    end
end


const POLEE_MODELS = Dict(
    "expression" => estimate_expression,
    "mixture"    => estimate_mixture
)