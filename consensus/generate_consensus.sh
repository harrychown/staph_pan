# This script is used to generate a consensus reference pan-genome
cluster_file="/data/OneDrive/bacterial_pangenome/outdata/non_ref_roary/clustered_proteins"
fasta_file="isolate.fasta"

# 1) SUBSET CLUSTER
line=$(head -n 1 $cluster_file)
echo $line | sed 's/[^ ]*//' | sed 's/ /\n/g' | sed '1d' > id.tmp
seqtk subseq $fasta_file id.tmp > subseq.tmp

# 2) PERFORM ALIGNMENT AND GENERATE CONSENSUS
muscle -in subseq.tmp -out aln.tmp -maxiters 1 -diags1
# Create HMM model
hmmbuild hmm.tmp aln.tmp
# Create consensus
hmmemit -c -o consensus.tmp hmm.tmp

# 3) RENAME AND SAVE CONSENSUS SEQUENCE
