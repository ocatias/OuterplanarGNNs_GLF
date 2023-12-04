declare -a datasets=("ogbg-molbace" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-moltoxcast")
declare -a configs=("cat_ogb_small.yaml" )

echo "HI"

# ZINC
python Exp/run_experiment.py -grid "Configs/Eval/cat_zinc.yaml" -dataset "ZINC" --candidates 48  --repeats 10 

# ogb
python Exp/run_experiment.py -grid Configs/Eval/cat_molhiv.yaml -dataset ogbg-molhiv --candidates 16 --repeats 10
for ds in "${datasets[@]}"
    do
    for config in "${configs[@]}"
        do
        echo "$ds"
        python Exp/run_experiment.py -grid "Configs/Eval/${config}" -dataset "$ds" --candidates 64  --repeats 10
        done
    done

