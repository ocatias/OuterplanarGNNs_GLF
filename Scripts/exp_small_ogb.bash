declare -a datasets=("ogbg-molbace" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-moltoxcast")

echo "Small OGB Experiments"

# ogb
for ds in "${datasets[@]}"
    do
    echo "$ds"
    python Exp/run_experiment.py -grid "Configs/No_Web/$1" -dataset "$ds" --candidates 64  --repeats 10
    done
    