# Script aims to extract pan-genome cluster sequences


cluster_file="/data/OneDrive/bacterial_pangenome/outdata/non_ref_roary/clustered_proteins"
gff_folder="/data/OneDrive/bacterial_pangenome/indata/non_ref_gff"
assembly_folder="/data/OneDrive/bacterial_pangenome/indata/assemblies"

fasta_file="${assembly_folder}/15-398-4_skesa.fa"
gff_file="${gff_folder}/15-398-4.gff"

# 1) REMOVE FASTA SEQUENCE FROM GFF
if grep -q "##FASTA" $gff_file
then 
	sed -n '/##FASTA/q;p' $gff_file | grep CDS > clean_gff.tmp
else
	grep CDS $gff_file > clean_gff.tmp
fi

# 2) REMOVE PREVIOUS ATTEMPTS
if [ -e isolate.fasta ]
then
	rm -rf isolate.fasta line.tmp
fi

# 2) UPDATE FASTA HEADERS
while read lines
do
	gene_id=$(echo "$lines" | cut -d$'\t' -f9 | cut -d"=" -f2 | cut -d";" -f1)
	# Replacing the name column with true name
	echo "$lines" | sed "s/CDS/${gene_id}/" >> line.tmp
done <clean_gff.tmp

# Extract sequence using bedtools
bedtools getfasta -name -fi $fasta_file -bed line.tmp > fasta.tmp
sed 's/::.*//g' fasta.tmp >> isolate.fasta

