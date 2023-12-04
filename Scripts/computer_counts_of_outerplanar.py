from Misc.run_converter import run


if __name__ == "__main__":
    n_outerplanar = dict()
    n_total = dict()
    for dataset in ['PCQM-Contact',
                    'ogbg-molhiv',
                    'peptides-func',
                    'zinc',
                    'ogbg-moltoxcast', 
                    'ogbg-moltox21', 
                    'ogbg-mollipo',
                    'ogbg-molbbbp',
                    'ogbg-molbace', 
                    'ogbg-molclintox', 
                    'ogbg-molsider',
                    'ogbg-molesol', 
]:
        print(f'\n##### {dataset} #####')
        n_outerplanar[dataset], n_total[dataset] = run({'--dataset': dataset})
    print('\\begin{tabular}{lrr}')
    print('\\toprule')
    print('Dataset & \#Graphs & Outerplanar \\\\')
    print('\midrule')
    for dataset in n_outerplanar:
        print(f'{dataset} & {n_total[dataset]} & {100*n_outerplanar[dataset]/n_total[dataset]:.2f} \\% \\\\')
    print('\\bottomrule')
    print('\\end{tabular}')