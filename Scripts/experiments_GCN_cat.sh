declare -a datasets=("ogbg-molbace" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-moltoxcast")
declare -a configs=("GCN_cat_ogb_small.yaml" )

echo "HI"

# ZINC
python Exp/run_experiment.py -grid "Configs/Eval_GCN/GCN_cat_zinc.yaml" -dataset "ZINC" --candidates 48  --repeats 10 


# ogb
for ds in "${datasets[@]}"
    do
    for config in "${configs[@]}"
        do
        echo "$ds"
        python Exp/run_experiment.py -grid "Configs/Eval_GCN/${config}" -dataset "$ds" --candidates 64  --repeats 10
        done
    done


python Exp/run_experiment.py -grid Configs/Eval_GCN/GCN_cat_molhiv.yaml -dataset ogbg-molhiv --candidates 16 --repeats 10
