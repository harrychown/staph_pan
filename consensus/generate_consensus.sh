# This script is used to generate a consensus reference pan-genome
cluster_file="/data/OneDrive/bacterial_pangenome/outdata/non_ref_roary/clustered_proteins"
fasta_file="isolate.fasta"

rm consensus_pangenome.fa

while read line
do
	# 1) SUBSET CLUSTER
	echo $line | sed 's/[^ ]*//' | sed 's/ /\n/g' | sed '1d' > id.tmp
	init_id=$(head -n 1 id.tmp)
	init_name=$(echo $line | cut -d ":" -f1)
	# Remove previous data
	seqtk subseq $fasta_file id.tmp > subseq.tmp

	# 2) PERFORM ALIGNMENT AND GENERATE CONSENSUS
	muscle -in subseq.tmp -out aln.tmp -maxiters 1 -diags1
	# Create HMM model
	hmmbuild hmm.tmp aln.tmp
	# Create consensus
	hmmemit -c -o consensus.tmp hmm.tmp
	
	# 3) RENAME AND SAVE CONSENSUS SEQUENCE
	sed "s/>aln-consensus/>$init_id $init_name/" consensus.tmp >> consensus_pangenome.fa
done <$cluster_file
