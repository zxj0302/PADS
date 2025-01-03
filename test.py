from src import create_heatmap

datasets = ['Brexit', 'Referendum_', 'Gun', 'Abortion', 'Election', 'Partisanship']
for d in datasets[:1]:
    create_heatmap(f'Output/{d}/graph.gml', f'Output/{d}/diffusion.json', save_path=f'figs/heatmap/{d}.pdf')