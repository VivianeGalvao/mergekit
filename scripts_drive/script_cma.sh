t='drive/MyDrive/artigo_vivi/resultados/cmaes_merged'
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
                    --opt_method CMA-ES \
                    --random-seed $i \
                    --storage-path $dir \
                    --force-population-size 10\
                    --max-fevals 500 \
                    mergekit/examples/evo_bert_large.yml

done
