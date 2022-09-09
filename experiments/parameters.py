from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


RESULT_PATH = Path('results', 'parameters.tsv')


def main():
    df = pd.read_csv(RESULT_PATH, sep='\t')
    df.loc[:, 'convergence'] = df['convergence'].apply(eval)

    # params = params.merge(pd.DataFrame({'pop-size': [128, 256, 512]}),
    #                       how='cross')
    # params = params.merge(pd.DataFrame({'pop-count': [2, 3, 4]}), how='cross')
    # params = params.merge(pd.DataFrame({'elite': [.10, .20, .30]}), how='cross')
    # params = params.merge(pd.DataFrame({'rhoe': [.60, .75, .90]}), how='cross')
    combinations = ['pop-size', 'pop-count', 'elite', 'rhoe']

    # FIXME Todos os cromosomos são recalculados, incluindo elites
    # TODO Testar com os GPU decoders

    ############################################################################
    # *** CPU: ***
    # Se diminui o rhoe e aumenta o elite, a solução piora
    #   => Mais elites mas o viés fica nos demais cromosomos
    #   => Dificuldade para "exploitation" ao passar das gerações
    #   => Menos cromosomos sendo atualizados

    # No. de elites: quantidades menores = melhor resultado
    # TODO testar com valores ainda menores

    # No. de populações: N/A
    # TODO testar com gaps maiores

    # Tamanho da população: N/A
    #   => Diferença deve surgir apenas no GPU decoder

    fixed = {
        # 'pop-size': 128,
        # 'pop-count': 2,
        # 'elite': .10,
        # 'rhoe': .75,
    }

    for param, value in fixed.items():
        df = df[df[param] == value]
        assert not df.empty, f"Couldn't find {param}={value}"

    fig, axes = plt.subplots(2, 2, figsize=(2 * 10.80, 2 * 7.20))
    axs = {problem: axes[i // 2, i % 2]
           for i, problem in enumerate(['tsp', 'scp', 'cvrp', 'cvrp_greedy'])}

    for problem, ax in axs.items():
        ax.set_title(problem)
        ax.grid()

    for _, row in df.iterrows():
        label = '-'.join(map(str, (row[c] for c in combinations)))
        convergence = row['convergence']
        fitness, elapsed = zip(*convergence)
        axs[row['problem']].plot(elapsed, fitness,
                                 label=label if row['problem'] == 'tsp' else '')

    fig.legend()
    fig.savefig('parameters.png', dpi=200)


if __name__ == '__main__':
    main()
