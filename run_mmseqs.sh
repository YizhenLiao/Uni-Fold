query=$1        # query csv / fasta / fastas. a3ms are not well supported.
dbbase=$2       # home dir of mmseqs databases.
result=$3       # to output feature.pkl.gz files.
a3m_out=$4      # to output raw a3m files and other output files of mmseqs.
templ_out=$5    # to output template.pkl.gz files.
threads=$6      # num workers.


python unifold/msa/mmseqs/mmseqs_search.py $query $dbbase $a3m_out --threads=$threads --remove-temp=1
python unifold/msa/mmseqs/make_template_features.py $a3m_out $dbbase $templ_out --num-workers=$threads
python unifold/msa/mmseqs/merge_features.py $a3m_out $templ_out $dbbase $result --num-workers=$threads
