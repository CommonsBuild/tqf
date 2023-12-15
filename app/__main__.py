import click
import pandas as pd


@click.command()
@click.option(
    '-r',
    '--round',
    'round',
    type=click.Choice(['tegr1', 'tegr2', 'tegr3', 'all'], case_sensitive=False),
)
def main(round: str):
    if round == 'tegr1':
        from app.tqf.tegr1 import tegr1_qf

        print(tegr1_qf.view_results())

    elif round == 'tegr2':
        from app.tqf.tegr2 import tegr2_qf

        print(tegr2_qf.view_results())

    elif round == 'tegr3':
        from app.tqf.tegr3 import tegr3_qf

        print(tegr3_qf.view_results())

    elif round == 'all':
        from app.tqf.tegr1 import tegr1_qf
        from app.tqf.tegr2 import tegr2_qf
        from app.tqf.tegr3 import tegr3_qf

        r1 = tegr1_qf.view_results()
        r2 = tegr2_qf.view_results()
        r3 = tegr3_qf.view_results()

        total = (
            pd.concat([r1, r2, r3])
            .groupby('Grant Name')
            .sum()
            .reset_index()
            .sort_values('Matching Funds Boosted', ascending=False)
        )

        print(total)


if __name__ == '__main__':
    main()
