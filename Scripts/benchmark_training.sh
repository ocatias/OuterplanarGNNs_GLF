declare -a datasets=("ZINC" "ogbg-molbace" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-moltoxcast" "ogbg-molhiv")
declare -a configs=("bench_cat.yaml" "bench_gin.yaml")

echo "Benmarking Training"

# ogb
for ds in "${datasets[@]}"
    do
    for config in "${configs[@]}"
        do
        echo "$ds"
        python Exp/run_experiment.py -grid "Configs/Benchmark/${config}" -dataset "$ds"  --repeats 5
        done
    done
