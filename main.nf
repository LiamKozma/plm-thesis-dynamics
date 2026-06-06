#!/usr/bin/env nextflow

// Parameter Channels
seeds_ch = Channel.fromList((0..<params.num_seeds).collect { it + params.starting_seed })
oracles_ch = Channel.fromList(params.oracle_architectures)
sigmas_ch = Channel.fromList(params.base_sigmas)
n_trains_ch = Channel.fromList(params.n_trains)
n_pools_ch = Channel.fromList(params.n_pools)
shifts_ch = Channel.fromList(params.shifts)
batch_sizes_ch = Channel.fromList(params.batch_sizes)
hidden_dims_ch = Channel.fromList(params.hidden_dims)

process GEN_SOURCE {
    tag "Src S:${seed} Shf:${shift_delta}"
    publishDir "${params.data_dir}/${params.dataset}/raw_data/", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_delta)

    output:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_delta), path("source_X_S${seed}_N${n_train}_Shf${shift_delta}.npy"), path("source_y_S${seed}_N${n_train}_Shf${shift_delta}.npy"), emit: data
    path "landscape_diagnostic_seed_${seed}.png", emit: diagnostics

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode source \
        --n_train ${n_train} \
        --shift ${shift_delta} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --n_classes ${params.n_classes} \
        --oracle_layers ${oracle} \
        --base_sigma ${sigma} \
        --seed ${seed}

    mv source_X.npy source_X_S${seed}_N${n_train}_Shf${shift_delta}.npy
    mv source_y.npy source_y_S${seed}_N${n_train}_Shf${shift_delta}.npy
    """
}

process TRAIN_SOURCE {
    tag "Train S:${seed} N:${n_train} Batch:${batch}"
    publishDir "${params.metrics_dir}/${params.dataset}/experiments/adapt/", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_delta), path(x_file), path(y_file), val(batch), val(h_dim)

    output:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_delta), path(x_file), path("model_S${seed}_N${n_train}_Shf${shift_delta}.pt"), val(batch), val(h_dim)

    script:
    """
    python ${projectDir}/src/train.py \
        --source_x ${x_file} \
        --source_y ${y_file} \
        --ref_x ${x_file} \
        --epochs ${params.base_epochs} \
        --batch_size ${batch} \
        --lr ${params.learning_rate} \
        --hidden_dim ${h_dim} \
        --dropout ${params.dropout} \
        --num_classes ${params.n_classes} \
        --output_model model_S${seed}_N${n_train}_Shf${shift_delta}.pt
    """
}

process GEN_TARGET {
    tag "Tgt S:${seed} P:${n_pool}"
    publishDir "${params.data_dir}/${params.dataset}/raw_data/", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_pool)

    output:
    tuple val(seed), val(oracle), val(sigma), val(n_pool), path("pool_X_S${seed}_NP${n_pool}.npy"), path("pool_y_S${seed}_NP${n_pool}.npy"), path("test_X_S${seed}_NP${n_pool}.npy"), path("test_y_S${seed}_NP${n_pool}.npy")

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode target \
        --n_pool ${n_pool} \
        --n_test ${params.n_test} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --n_classes ${params.n_classes} \
        --oracle_layers ${oracle} \
        --base_sigma ${sigma} \
        --seed ${seed}

    mv target_pool_X.npy pool_X_S${seed}_NP${n_pool}.npy
    mv target_pool_y.npy pool_y_S${seed}_NP${n_pool}.npy
    mv target_test_X.npy test_X_S${seed}_NP${n_pool}.npy
    mv target_test_y.npy test_y_S${seed}_NP${n_pool}.npy
    """
}

process TEST_ADAPTATION {
    tag "Adapt S:${seed} NP:${n_pool} B:${batch} H:${h_dim}"
    publishDir "${params.metrics_dir}/${params.dataset}/experiments/adapt/", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_delta), path(ref_x), path(source_model), val(batch), val(h_dim), val(n_pool), path(pool_x), path(pool_y), path(test_x), path(test_y)

    output:
    path "adapt_log_S${seed}_N${n_train}_NP${n_pool}_Shf${shift_delta}_B${batch}_H${h_dim}.log"
    path "*_batch_log.csv", optional: true
    path "adapted_model_S${seed}_N${n_train}_NP${n_pool}_Shf${shift_delta}_B${batch}_H${h_dim}.pt"

    script:
    """
    python ${projectDir}/src/${params.adapt_script} \
        --base_model ${source_model} \
        --pool_x ${pool_x} \
        --pool_y ${pool_y} \
        --test_x ${test_x} \
        --test_y ${test_y} \
        --ref_x ${ref_x} \
        --batch_size ${batch} \
        --lr ${params.adapt_lr} \
        --hidden_dim ${h_dim} \
        --dropout ${params.dropout} \
        --num_classes ${params.n_classes} \
        --output_model adapted_model_S${seed}_N${n_train}_NP${n_pool}_Shf${shift_delta}_B${batch}_H${h_dim}.pt \
        > adapt_log_S${seed}_N${n_train}_NP${n_pool}_Shf${shift_delta}_B${batch}_H${h_dim}.log
    """
}

workflow {
    // Source Generation + Training Sweep
    source_params = seeds_ch.combine(oracles_ch).combine(sigmas_ch).combine(n_trains_ch).combine(shifts_ch)
    GEN_SOURCE(source_params)
    sources = GEN_SOURCE.out.data

    train_params = sources.combine(batch_sizes_ch).combine(hidden_dims_ch)
    source_models = TRAIN_SOURCE(train_params)

    // Target Generation
    target_params = seeds_ch.combine(oracles_ch).combine(sigmas_ch).combine(n_pools_ch)
    targets = GEN_TARGET(target_params)

    // Cross-Product Engine
    // join on (seed, oracle, sigma) — enforces matched source/target environments
    experiments = source_models.combine(targets, by: [0, 1, 2])

    // Dispatch
    if (params.mode == 'adapt') {
        TEST_ADAPTATION(experiments)
    } else {
        error "Invalid mode specified in YAML. Choose 'adapt'."
    }
}
