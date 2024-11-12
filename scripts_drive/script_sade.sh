t='drive/MyDrive/artigo_vivi/resultados/sade_merged'
mkdir $t
for i in $(seq 1 5);
do
    dir=$t"/merge_"$i
    mkdir $dir
    echo $dir

    mergekit-evolve --batch-size 10 \
                    --no-in-memory \
                    --allow-crimes \
                    --no-reshard \
                    --strategy pool \
                    --opt_method SaDE \
                    --random-seed $i \
                    --storage-path $dir \
                    --force-population-size 10\
                    --max-fevals 500 \
                    mergekit/examples/evo_bert_large.yml
done
