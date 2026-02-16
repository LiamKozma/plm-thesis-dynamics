#!/usr/bin/env nextflow

// DEFAULT PARAMS
params.mode = 'adapt'       // 'adapt' (train) or 'eval' (inference)
params.data = 'manifold'    // 'manifold' (Your View) or 'highdim' (Prof View)

// 1. Generate Source Data
process GEN_SOURCE {
    publishDir "results/${params.data}/data/source", mode: 'copy' // Separate folders!

    output:
        path "source_X.npy", emit: x
        path "source_y.npy", emit: y

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --type ${params.data} \
        --shift 0.0 \
        --output_x source_X.npy \
        --output_y source_y.npy
    """
}

// 2. Train Source Model
process TRAIN_SOURCE {
    publishDir "results/${params.data}/models", mode: 'copy'

    input:
        path x_file
        path y_file
    output:
        path "source_model.pt", emit: model
    script:
    """
    python ${projectDir}/src/train.py \
        --source_x ${x_file} --source_y ${y_file} --output_model source_model.pt
    """
}

// 3. Generate Targets
process GEN_TARGET {
    tag "Shift: ${shift_val}" 
    input:
        val shift_val
    output:
        tuple val(shift_val), path("target_X.npy"), path("target_y.npy")
    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --type ${params.data} \
        --shift ${shift_val} \
        --output_x target_X.npy \
        --output_y target_y.npy
    """
}

// OPTION A: Adaptation (Train)
process TEST_ADAPTATION {
    tag "Adapt [${params.data}] Shift: ${shift_val}"
    publishDir "results/${params.data}/experiments/adapt", mode: 'copy'

    input:
    tuple val(shift_val), path(tgt_x), path(tgt_y)

    output:
    path "adapt_result_${shift_val}.log"

    script:
    """
    python ${projectDir}/src/train.py \
        --source_x ${tgt_x} \
        --source_y ${tgt_y} \
        --output_model adapted_model.pt > adapt_result_${shift_val}.log
    """
}

// OPTION B: Generalization (Eval)
process TEST_GENERALIZATION {
    tag "Eval [${params.data}] Shift: ${shift_val}"
    publishDir "results/${params.data}/experiments/eval", mode: 'copy'

    input:
    path model_file
    tuple val(shift_val), path(tgt_x), path(tgt_y)

    output:
    path "eval_result_${shift_val}.log"

    script:
    """
    python ${projectDir}/src/eval.py \
        --model_path ${model_file} \
        --target_x ${tgt_x} \
        --target_y ${tgt_y} > eval_result_${shift_val}.log
    """
}

workflow {
    source_ch = GEN_SOURCE()
    model_ch = TRAIN_SOURCE(source_ch.x, source_ch.y)
    
    shifts = Channel.fromList([0.5, 1.0, 1.5, 2.0, 3.0])
    targets_ch = GEN_TARGET(shifts)

    if (params.mode == 'adapt') {
        TEST_ADAPTATION(targets_ch)
    } else {
        TEST_GENERALIZATION(model_ch.model, targets_ch)
    }
}
