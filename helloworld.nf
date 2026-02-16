#!/usr/bin/env nextflow

process SAY_HELLO {
    // Publish the output to your S3 Data Lake
    publishDir "${params.outdir}/hello_test", mode: 'copy'

    output:
    path "hello.txt"

    script:
    """
    echo "Hybrid Cloud Integration Successful!" > hello.txt
    # We must escape the \$ so Nextflow doesn't try to read it
    echo "Timestamp: \$(date)" >> hello.txt
    echo "Runner: \$USER on \$(hostname)" >> hello.txt
    """
}

workflow {
    SAY_HELLO()
}
