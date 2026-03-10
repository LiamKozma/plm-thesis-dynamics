graph TD
    subgraph Configuration
        YAML[adapt_wasserstein.yaml<br/>Hyperparameters & Experimental Matrix] --> NF[main.nf<br/>Nextflow Orchestrator]
    end

    subgraph Phase 1: Source Pipeline
        NF -->|seed, n_train| GEN_SRC[generate_simulation.py<br/>Mode: Source]
        GEN_SRC -->|source_X.npy, source_y.npy| TRAIN[train.py]
        TRAIN -.->|Imports| MODEL[model.py: FitnessPredictor]
        TRAIN -.->|Imports| METRICS[metrics.py: rBME, Wasserstein]
        TRAIN -->|source_model.pt| ADAPT
        TRAIN -->|source_model.pt| EVAL
    end

    subgraph Phase 2: Target Pipeline
        NF -->|seed, n_pool, shift| GEN_TGT[generate_simulation.py<br/>Mode: Target]
        GEN_TGT -->|tgt_pool_X.npy, tgt_pool_y.npy| ADAPT
        GEN_TGT -->|tgt_test_X.npy, tgt_test_y.npy| ADAPT
        GEN_TGT -->|tgt_test_X.npy, tgt_test_y.npy| EVAL
    end

    subgraph Phase 3: Execution Engine
        ADAPT[adapt.py<br/>Active Learning Loop] -.->|Imports| MODEL
        ADAPT -.->|Imports| METRICS
        ADAPT -->|batch_log.csv, adapt.log| RES
        
        EVAL[eval.py<br/>Generalization Testing] -.->|Imports| MODEL
        EVAL -.->|Imports| METRICS
        EVAL -->|eval.log| RES
    end

    subgraph Phase 4: Analysis
        RES[view_results.py<br/>Log Parser] --> PLOTS[Crash Plots & Recovery Curves]
    end
