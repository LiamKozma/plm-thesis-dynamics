#!/usr/bin/env nextflow

// 1. Generate the "Source" Domain (The Wild Type)
process GEN_SOURCE {
    publishDir 'results/data/source', mode: 'copy'

    output:
    path "source_X.npy", emit: x
    path "source_y.npy", emit: y

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --shift 0.0 \
        --output_x source_X.npy \
        --output_y source_y.npy
    """
}

// 2. Train the Baseline Model on Source
process TRAIN_SOURCE {
    publishDir 'results/models', mode: 'copy'

    input:
    path x_file
    path y_file

    output:
    path "source_model.pt", emit: model

    script:
    """
    python ${projectDir}/src/train.py \
        --source_x ${x_file} \
        --source_y ${y_file} \
        --output_model source_model.pt
    """
}

// 3. Generate "Target" Domains (The Mutations/Shifts)
// We input a "shift_val" and output a specific dataset
process GEN_TARGET {
    tag "Shift: ${shift_val}" // Nice label in the logs

    input:
    val shift_val

    output:
    tuple val(shift_val), path("target_X.npy"), path("target_y.npy")

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --shift ${shift_val} \
        --output_x target_X.npy \
        --output_y target_y.npy
    """
}

// 4. Evaluate (The "Crash" Test)
// We haven't written eval.py yet, so we will reuse train.py 
// but just check the validation score on the new data
process TEST_GENERALIZATION {
    tag "Testing on Shift: ${shift_val}"
    publishDir 'results/experiments', mode: 'copy'

    input:
    path model_file
    tuple val(shift_val), path(tgt_x), path(tgt_y)

    output:
    path "result_${shift_val}.log"

    script:
    """
    echo "Evaluating Source Model on Shift ${shift_val}"
    python ${projectDir}/src/eval.py \
        --model_path ${model_file} \
        --target_x ${tgt_x} \
        --target_y ${tgt_y} > result_${shift_val}.log
    """
}

workflow {
    // A. Create Source Data
    source_ch = GEN_SOURCE()

    // B. Train Model on Source
    // CRITICAL: Connect the output of A to the input of B
    model_ch = TRAIN_SOURCE(source_ch.x, source_ch.y)

    // C. Define our Experiment Range
    shifts = Channel.fromList([0.5, 1.0, 1.5, 2.0, 3.0])

    // D. Run Parallel Generation
    targets_ch = GEN_TARGET(shifts)

    // E. Run Parallel Testing
    TEST_GENERALIZATION(model_ch.model, targets_ch)
}
