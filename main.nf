#!/usr/bin/env nextflow

// 1. Setup Channels from config.yaml arrays
seeds_ch = Channel.fromList((0..<params.num_seeds).collect { it + params.starting_seed })
n_trains_ch = Channel.fromList(params.n_trains)
n_pools_ch = Channel.fromList(params.n_pools)
shifts_ch = Channel.fromList(params.shifts)

// 2. Processes
process GEN_SOURCE {
    tag "Src N:${n_train} Shf:${shift_val} S:${seed}"
    publishDir "${params.outdir}/${params.dataset}/data/source", mode: 'copy'
    
    input:
    tuple val(seed), val(n_train), val(shift_val)

    output:
    tuple val(seed), val(n_train), val(shift_val), path("source_X_${n_train}_shf${shift_val}_s${seed}.npy"), path("source_y_${n_train}_shf${shift_val}_s${seed}.npy")

    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode source \
        --n_train ${n_train} \
        --shift ${shift_val} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --n_classes ${params.n_classes} \
        --oracle_layers ${params.oracle_layers.join(',')} \
        --seed ${seed}
    
    mv source_train_X.npy source_X_${n_train}_shf${shift_val}_s${seed}.npy
    mv source_train_y.npy source_y_${n_train}_shf${shift_val}_s${seed}.npy
    """
}

process TRAIN_SOURCE {
    tag "Train N:${n_train} Shf:${shift_val} S:${seed}"
    publishDir "${params.outdir}/${params.dataset}/models", mode: 'copy'

    input:
    tuple val(seed), val(n_train), val(shift_val), path(x_file), path(y_file)
    
    output:
    tuple val(seed), val(n_train), val(shift_val), path(x_file), path("model_N${n_train}_shf${shift_val}_s${seed}.pt")
    
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
        --num_classes ${params.n_classes} \
        --output_model model_N${n_train}_shf${shift_val}_s${seed}.pt
    """
}

process GEN_TARGET {
    tag "Tgt P:${n_pool} S:${seed}"
    publishDir "${params.outdir}/${params.dataset}/data/target", mode: 'copy'

    input:
    tuple val(seed), val(n_pool)
    
    output:
    tuple val(seed), val(n_pool), path("tgt_pool_X_${n_pool}_s${seed}.npy"), path("tgt_pool_y_${n_pool}_s${seed}.npy"), path("tgt_test_X_${n_pool}_s${seed}.npy"), path("tgt_test_y_${n_pool}_s${seed}.npy")
    
    script:
    """
    python ${projectDir}/src/generate_simulation.py \
        --mode target \
        --n_pool ${n_pool} \
        --n_test ${params.n_test} \
        --dim ${params.dim} \
        --n_families ${params.n_families} \
        --n_classes ${params.n_classes} \
        --oracle_layers ${params.oracle_layers.join(',')} \
        --seed ${seed}

    mv target_pool_X.npy tgt_pool_X_${n_pool}_s${seed}.npy
    mv target_pool_y.npy tgt_pool_y_${n_pool}_s${seed}.npy
    mv target_test_X.npy tgt_test_X_${n_pool}_s${seed}.npy
    mv target_test_y.npy tgt_test_y_${n_pool}_s${seed}.npy
    """
}

process TEST_ADAPTATION {
    tag "Adapt S:${seed} NTr:${n_train} NP:${n_pool} Shf:${shift_val}"
    publishDir "${params.outdir}/${params.dataset}/experiments/adapt", mode: 'copy'


    input:
    // Updated tuple order based on the new 'by: 0' combine logic
    tuple val(seed), val(n_train), val(shift_val), path(ref_x), path(source_model), val(n_pool), path(pool_x), path(pool_y), path(test_x), path(test_y)

    output:
    path "adapt_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log"
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
        --batch_size ${params.batch_size} \
        --lr ${params.learning_rate} \
        --hidden_dim ${params.hidden_dim} \
        --dropout ${params.dropout} \
        --num_classes ${params.n_classes} \
        --output_model adapted_model_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.pt \
        > adapt_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log
    """
}

process TEST_GENERALIZATION {
    tag "Eval S:${seed} NTr:${n_train} NP:${n_pool} Shf:${shift_val}"
    publishDir "${params.outdir}/${params.dataset}/experiments/eval", mode: 'copy'


    input:
    tuple val(seed), val(n_train), val(shift_val), path(ref_x), path(source_model), val(n_pool), path(pool_x), path(pool_y), path(test_x), path(test_y)

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
        --num_classes ${params.n_classes} \
        --batch_size ${params.batch_size} > eval_NTr${n_train}_NP${n_pool}_Shf${shift_val}_s${seed}.log
    """
}

workflow {
    // Phase 1: Pre-training (Source gets the Shift parameter!)
    source_params = seeds_ch.combine(n_trains_ch).combine(shifts_ch)
    sources = GEN_SOURCE(source_params) 
    source_models = TRAIN_SOURCE(sources)

    // Phase 2: Target Data Generation (Target is constant, no Shift parameter needed)
    target_params = seeds_ch.combine(n_pools_ch)
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
