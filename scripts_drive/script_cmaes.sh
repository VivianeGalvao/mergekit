export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

rm resultados/training_log.csv
t='resultados/cmaes_merged'
mkdir $t
for i in $(seq 1 3);
do
    dir=$t"/merge_"$i
    mkdir $dir
    echo $dir

    mergekit-evolve --batch-size 1 \
                    --no-in-memory \
                    --allow-crimes \
                    --no-reshard \
                    --strategy buffered \
                    --num-gpus 1 \
                    --random-seed $i \
                    --storage-path $dir \
                    --force-population-size 5\
                    --max-fevals 100 \
                    mergekit/examples/evo_qwen_reasoning.yml

    rm resultados/training_log.csv

done