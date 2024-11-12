t='drive/MyDrive/artigo_vivi/resultados/sade_merged'
mkdir $t
i=1
dir=$t"/merge_"$i
mkdir $dir


mergekit-evolve --batch-size 5 \
                --no-in-memory \
                --allow-crimes \
                --no-reshard \
                --strategy pool \
                --opt_method SaDE \
                --random-seed $i \
                --storage-path $dir \
                --force-population-size 5\
                --max-fevals 5 \
                mergekit/examples/evo_bert_large.yml