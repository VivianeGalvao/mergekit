export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

t='drive/MyDrive/artigo_vivi/taic/resultados/cmaes_merged'
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
                    --n_gpus 1 \
                    --load-in-8bit \
                    --random-seed $i \
                    --storage-path $dir \
                    --force-population-size 5\
                    --max-fevals 10 \
                    mergekit/examples/evo_qwen_reasoning.yml

done