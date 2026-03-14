#!/usr/bin/env nextflow

// 1. Setup Channels from config.yaml arrays
seeds_ch = Channel.fromList((0..<params.num_seeds).collect { it + params.starting_seed })
oracles_ch = Channel.fromList(params.oracle_architectures)
sigmas_ch = Channel.fromList(params.base_sigmas)
n_trains_ch = Channel.fromList(params.n_trains)
n_pools_ch = Channel.fromList(params.n_pools)
shifts_ch = Channel.fromList(params.shifts)
batch_sizes_ch = Channel.fromList(params.batch_sizes)
hidden_dims_ch = Channel.fromList(params.hidden_dims)

// 2. Processes
process GEN_SOURCE {
    tag "Src S:${seed} Shf:${shift_val}"
    // Route heavy data to /scratch
    publishDir "${params.data_dir}/${params.dataset}/data/source", mode: 'copy'
    
    input:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_val)

    output:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_val), path("source_X.npy"), path("source_y.npy")

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode source \
        --n_train ${n_train} \
        --shift ${shift_val} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --n_classes ${params.n_classes} \
        --oracle_layers ${oracle} \
        --base_sigma ${sigma} \
        --seed ${seed}
    
    mv source_train_X.npy source_X.npy
    mv source_train_y.npy source_y.npy
    """
}

process TRAIN_SOURCE {
    tag "Train S:${seed} N:${n_train} Batch:${batch}"
    // Route metrics and models to /work
    publishDir "${params.metrics_dir}/${params.dataset}/models", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_val), path(x_file), path(y_file), val(batch), val(h_dim)
    
    output:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_val), path(x_file), path("model_S${seed}.pt"), val(batch), val(h_dim)
    
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
        --output_model model_S${seed}.pt
    """
}

process GEN_TARGET {
    tag "Tgt S:${seed} P:${n_pool}"
    // Route heavy data to /scratch
    publishDir "${params.data_dir}/${params.dataset}/data/target", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_pool)
    
    output:
    tuple val(seed), val(oracle), val(sigma), val(n_pool), path("pool_X.npy"), path("pool_y.npy"), path("test_X.npy"), path("test_y.npy")
    
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

    mv target_pool_X.npy pool_X.npy
    mv target_pool_y.npy pool_y.npy
    mv target_test_X.npy test_X.npy
    mv target_test_y.npy test_y.npy
    """
}

process TEST_ADAPTATION {
    tag "Adapt S:${seed} NP:${n_pool} B:${batch} H:${h_dim}"
    // Route metrics and models to /work
    publishDir "${params.metrics_dir}/${params.dataset}/experiments/adapt", mode: 'copy'

    input:
    tuple val(seed), val(oracle), val(sigma), val(n_train), val(shift_val), path(ref_x), path(source_model), val(batch), val(h_dim), val(n_pool), path(pool_x), path(pool_y), path(test_x), path(test_y)

    output:
    path "adapt_log_S${seed}_NP${n_pool}_Shf${shift_val}_B${batch}_H${h_dim}.log"
    path "*_batch_log.csv", optional: true
    
    script:
    """
    python ${projectDir}/src/adapt.py \
        --base_model ${source_model} \
        --pool_x ${pool_x} \
        --pool_y ${pool_y} \
        --test_x ${test_x} \
        --test_y ${test_y} \
        --ref_x ${ref_x} \
        --batch_size ${batch} \
        --lr ${params.learning_rate} \
        --hidden_dim ${h_dim} \
        --dropout ${params.dropout} \
        --num_classes ${params.n_classes} \
        --output_model adapted_model_S${seed}_NP${n_pool}_Shf${shift_val}_B${batch}_H${h_dim}.pt \
        > adapt_log_S${seed}_NP${n_pool}_Shf${shift_val}_B${batch}_H${h_dim}.log
    """
}

workflow {
    // Phase 1: Pre-training (Combines data params)
    source_params = seeds_ch.combine(oracles_ch).combine(sigmas_ch).combine(n_trains_ch).combine(shifts_ch)
    sources = GEN_SOURCE(source_params) 
    
    // Add Neural Network sweeping params to the training phase
    train_params = sources.combine(batch_sizes_ch).combine(hidden_dims_ch)
    source_models = TRAIN_SOURCE(train_params)

    // Phase 2: Target Data Generation
    target_params = seeds_ch.combine(oracles_ch).combine(sigmas_ch).combine(n_pools_ch)
    targets = GEN_TARGET(target_params)

    // Phase 3: The Cross-Product Engine
    // Join on Seed (0), Oracle (1), and Sigma (2) so source and target environments match
    experiments = source_models.combine(targets, by: [0, 1, 2])

    // Phase 4: Execution
    if (params.mode == 'adapt') {
        TEST_ADAPTATION(experiments)
    } else {
        error "Invalid mode specified in YAML. Choose 'adapt'."
    }
}
