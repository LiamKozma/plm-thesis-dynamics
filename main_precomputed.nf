#!/usr/bin/env nextflow

// =============================================================================
// Real-data variant of main.nf: the data generator (GEN_SOURCE/GEN_TARGET) is
// REPLACED by precomputed real ESM-2 embeddings produced by
// src/precompute_real_embeddings.py. We only run training + adaptation, so the
// dip / recovery-threshold experiment uses identical machinery to the synthetic
// runs -- only the data changed. See HOW_TO_SEE_THE_DIP.md.
//
// Sweeps params.shifts (OOD fraction r) x params.seeds (for error bars). The
// embeddings are deterministic, so seeds vary only model init + pool shuffle
// order -- which is what produces the SGD noise floor you need to bound.
//
// Expects, in params.precomputed_dir, one set per shift r:
//   source_Shf${r}_X.npy  source_Shf${r}_y.npy
//   pool_Shf${r}_X.npy    pool_Shf${r}_y.npy
//   test_Shf${r}_X.npy    test_Shf${r}_y.npy
//
// IMPORTANT: set params.num_classes to dataset_info.json -> num_classes.
// =============================================================================

process TRAIN_SOURCE {
    tag "Train Shf:${r} S:${seed}"
    publishDir "${params.metrics_dir}/${params.dataset}/experiments/adapt/", mode: 'copy'

    input:
    tuple val(r), path(src_x), path(src_y), path(pool_x), path(pool_y), path(test_x), path(test_y), val(seed)

    output:
    tuple val(r), val(seed), path(src_x), path("model_Shf${r}_S${seed}.pt"), path(pool_x), path(pool_y), path(test_x), path(test_y)

    script:
    """
    python ${projectDir}/src/train.py \
        --source_x ${src_x} \
        --source_y ${src_y} \
        --ref_x ${src_x} \
        --epochs ${params.base_epochs} \
        --batch_size ${params.batch_size} \
        --lr ${params.learning_rate} \
        --hidden_dim ${params.hidden_dim} \
        --dropout ${params.dropout} \
        --num_classes ${params.num_classes} \
        --seed ${seed} \
        --output_model model_Shf${r}_S${seed}.pt
    """
}

process TEST_ADAPTATION {
    tag "Adapt Shf:${r} S:${seed}"
    publishDir "${params.metrics_dir}/${params.dataset}/experiments/adapt/", mode: 'copy'

    input:
    tuple val(r), val(seed), path(ref_x), path(base_model), path(pool_x), path(pool_y), path(test_x), path(test_y)

    output:
    path "adapt_log_Shf${r}_S${seed}.log"
    path "*_batch_log.csv", optional: true
    path "adapted_model_Shf${r}_S${seed}.pt"

    script:
    """
    python ${projectDir}/src/${params.adapt_script} \
        --base_model ${base_model} \
        --pool_x ${pool_x} \
        --pool_y ${pool_y} \
        --test_x ${test_x} \
        --test_y ${test_y} \
        --ref_x ${ref_x} \
        --batch_size ${params.adapt_batch_size} \
        --lr ${params.adapt_lr} \
        --hidden_dim ${params.hidden_dim} \
        --dropout ${params.dropout} \
        --num_classes ${params.num_classes} \
        --eval_every ${params.eval_every} \
        --seed ${seed} \
        --output_model adapted_model_Shf${r}_S${seed}.pt \
        > adapt_log_Shf${r}_S${seed}.log
    """
}

workflow {
    sets = Channel.fromList(params.shifts).map { r ->
        def d = params.precomputed_dir
        tuple(
            r,
            file("${d}/source_Shf${r}_X.npy"),
            file("${d}/source_Shf${r}_y.npy"),
            file("${d}/pool_Shf${r}_X.npy"),
            file("${d}/pool_Shf${r}_y.npy"),
            file("${d}/test_Shf${r}_X.npy"),
            file("${d}/test_Shf${r}_y.npy"),
        )
    }

    // cartesian product: every (shift, seed) pair is one train+adapt run
    work = sets.combine(Channel.fromList(params.seeds))

    trained = TRAIN_SOURCE(work)
    TEST_ADAPTATION(trained)
}
