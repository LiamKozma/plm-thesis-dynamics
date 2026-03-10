#!/usr/bin/env nextflow

// 1. Setup Channels from config.yaml arrays
seeds_ch = Channel.fromList((0..<params.num_seeds).collect { it + params.starting_seed })
n_trains_ch = Channel.fromList(params.n_trains)
n_pools_ch = Channel.fromList(params.n_pools)
shifts_ch = Channel.fromList(params.shifts)

// 2. Processes
process GEN_SOURCE {
    tag "Src N:${n_train} Seed:${seed}"
    publishDir "results/${params.dataset}/data/source", mode: 'copy'
    
    input:
    tuple val(seed), val(n_train)

    output:
    tuple val(seed), val(n_train), path("source_X_${n_train}_s${seed}.npy"), path("source_y_${n_train}_s${seed}.npy")

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode source \
        --n_train ${n_train} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --seed ${seed}
    
    mv source_train_X.npy source_X_${n_train}_s${seed}.npy
    mv source_train_y.npy source_y_${n_train}_s${seed}.npy
    """
}

process TRAIN_SOURCE {
    tag "Train N:${n_train} Seed:${seed}"
    publishDir "results/${params.dataset}/models", mode: 'copy'

    input:
    tuple val(seed), val(n_train), path(x_file), path(y_file)
    
    output:
    tuple val(seed), val(n_train), path(x_file), path("model_N${n_train}_s${seed}.pt")
    
    script:
    """
    python ${projectDir}/src/train.py \
        --source_x ${x_file} \
        --source_y ${y_file} \
        --ref_x ${x_file} \
        --epochs ${params.base_epochs} \
        --batch_size ${params.batch_size} \
        --lr ${params.learning_rate} \
        --hidden_dim ${params.hidden_dim} \
        --dropout ${params.dropout} \
        --output_model model_N${n_train}_s${seed}.pt
    """
}

process GEN_TARGET {
    tag "Tgt P:${n_pool} Shf:${shift_val} Seed:${seed}"
    publishDir "results/${params.dataset}/data/target", mode: 'copy'
    
    input:
    tuple val(seed), val(n_pool), val(shift_val)
    
    output:
    tuple val(seed), val(n_pool), val(shift_val), path("tgt_pool_X_${n_pool}_${shift_val}_s${seed}.npy"), path("tgt_pool_y_${n_pool}_${shift_val}_s${seed}.npy"), path("tgt_test_X_${n_pool}_${shift_val}_s${seed}.npy"), path("tgt_test_y_${n_pool}_${shift_val}_s${seed}.npy")
    
    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode target \
        --n_pool ${n_pool} \
        --n_test ${params.n_test} \
        --shift ${shift_val} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --seed ${seed}

    mv target_pool_X.npy tgt_pool_X_${n_pool}_${shift_val}_s${seed}.npy
    mv target_pool_y.npy tgt_pool_y_${n_pool}_${shift_val}_s${seed}.npy
    mv target_test_X.npy tgt_test_X_${n_pool}_${shift_val}_s${seed}.npy
    mv target_test_y.npy tgt_test_y_${n_pool}_${shift_val}_s${seed}.npy
    """
}

process TEST_ADAPTATION {
    tag "Adapt [${params.dataset}] S:${seed} NTr:${n_train} NP:${n_pool} Shf:${shift_val}"
    publishDir "results/${params.dataset}/experiments/adapt", mode: 'copy'

    input:
    // The 'by: 0' cross-join feeds this combined tuple:
    tuple val(seed), val(n_train), path(ref_x), path(source_model), val(n_pool), val(shift_val), path(pool_x), path(pool_y), path(test_x), path(test_y)

    output:
    path "adapt_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log"
    path "*_batch_log.csv", optional: true // Captures the batch-by-batch metrics!

    script:
    """
    python ${projectDir}/src/adapt.py \
        --base_model ${source_model} \
        --pool_x ${pool_x} \
        --pool_y ${pool_y} \
        --test_x ${test_x} \
        --test_y ${test_y} \
        --ref_x ${ref_x} \
        --batch_size ${params.batch_size} \
        --lr ${params.learning_rate} \
        --hidden_dim ${params.hidden_dim} \
        --dropout ${params.dropout} \
        --output_model adapted_model_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.pt \
        > adapt_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log
    """
}

process TEST_GENERALIZATION {
    tag "Eval [${params.dataset}] S:${seed} NTr:${n_train} NP:${n_pool} Shf:${shift_val}"
    publishDir "results/${params.dataset}/experiments/eval", mode: 'copy'

    input:
    tuple val(seed), val(n_train), path(ref_x), path(source_model), val(n_pool), val(shift_val), path(pool_x), path(pool_y), path(test_x), path(test_y)

    output:
    path "eval_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log"

    script:
    """
    python ${projectDir}/src/eval.py \
        --model_path ${source_model} \
        --ref_x ${ref_x} \
        --target_x ${test_x} \
        --target_y ${test_y} \
        --hidden_dim ${params.hidden_dim} \
        --dropout ${params.dropout} \
        --batch_size ${params.batch_size} > eval_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log
    """
}


workflow {
    // Phase 1: Pre-training (Source)
    source_params = seeds_ch.combine(n_trains_ch)
    sources = GEN_SOURCE(source_params) 
    source_models = TRAIN_SOURCE(sources)

    // Phase 2: Target Data Generation
    target_params = seeds_ch.combine(n_pools_ch).combine(shifts_ch)
    targets = GEN_TARGET(target_params)

    // Phase 3: The Cross-Product Engine
    experiments = source_models.combine(targets, by: 0)

    // Phase 4: Execution Routing
    if (params.mode == 'adapt') {
        TEST_ADAPTATION(experiments)
    } else if (params.mode == 'eval') {
        TEST_GENERALIZATION(experiments)
    } else {
        error "Invalid mode specified in YAML. Choose 'adapt' or 'eval'."
    }
}
