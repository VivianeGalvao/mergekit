t='drive/MyDrive/artigo_vivi/taic/resultados/cmaes_merged'
mkdir $t
for i in $(seq 1 3);
do
    dir=$t"/merge_"$i
    mkdir $dir
    echo $dir

    mergekit-evolve --batch-size 5 \
                    --no-in-memory \
                    --allow-crimes \
                    --no-reshard \
                    --strategy pool \
                    --random-seed $i \
                    --storage-path $dir \
                    --force-population-size 5\
                    --max-fevals 10 \
                    mergekit/examples/evo_qwen_reasoning.yml

done