import click
import pandas as pd
import panel as pn

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)


def pn_exception_handler(ex):
    ic("🔥🔥🔥")
    logging.error("Error", exc_info=ex)
    pn.state.notifications.send(
        "Error: %s" % ex, duration=int(10e3), background="black"
    )


pn.extension(
    "tabulator",
    "mathjax",
    exception_handler=pn_exception_handler,
    notifications=True,
    loading_spinner="dots",
    loading_color="#00aa41",
)
pn.state.notifications.position = "top-right"
pn.config.throttled = True


@click.command()
@click.option(
    "-r",
    "--round",
    "round",
    type=click.Choice(["tegr1", "tegr2", "tegr3", "all"], case_sensitive=False),
)
@click.option(
    "-c",
    "--cli",
    "cli",
    default=False,
    is_flag=True,
    help="Display results in cli rather than launching web application",
)
@click.option(
    "-a",
    "--admin",
    "admin",
    default=False,
    is_flag=True,
    help="Whether to launch the admin panel with the app.",
)
@click.option(
    "-p",
    "--port",
    "port",
    default=5006,
    type=int,
    help="Port number to launch app on.",
)
def main(round: str, cli: bool, admin: bool, port: int):
    if round == "tegr1":
        if cli:
            from tqf.tegr.tegr1 import tegr1_qf

            print(tegr1_qf.view_results(tabulator=False, to_csv=True))
        else:
            from tqf.tegr.tegr1 import tegr1_app

            pn.serve(tegr1_app.servable(), port=port, show=False, admin=admin)

    elif round == "tegr2":
        if cli:
            from tqf.tegr.tegr2 import tegr2_qf

            print(tegr2_qf.view_results(tabulator=False, to_csv=True))
        else:
            from tqf.tegr.tegr2 import tegr2_app

            pn.serve(tegr2_app.servable(), port=port, show=False, admin=admin)

    elif round == "tegr3":
        if cli:
            from tqf.tegr.tegr3 import tegr3_qf

            print(tegr3_qf.view_results(tabulator=False, to_csv=True))
        else:
            from tqf.tegr.tegr3 import tegr3_app

            pn.serve(tegr3_app.servable(), port=port, show=False, admin=admin)

    elif round == "all":
        if cli:
            from tqf.tegr.tegr1 import tegr1_qf
            from tqf.tegr.tegr2 import tegr2_qf
            from tqf.tegr.tegr3 import tegr3_qf

            r1 = tegr1_qf.view_results(tabulator=False, to_csv=True)
            r2 = tegr2_qf.view_results(tabulator=False, to_csv=True)
            r3 = tegr3_qf.view_results(tabulator=False, to_csv=True)

            total = (
                pd.concat([r1, r2, r3])
                .groupby("Grant Name")
                .sum()
                .reset_index()
                .sort_values("Matching Funds Boosted", ascending=False)
            )

            print(total)

        else:
            from tqf.tegr.tegr1 import tegr1_app
            from tqf.tegr.tegr2 import tegr2_app
            from tqf.tegr.tegr3 import tegr3_app

            pn.serve(
                {
                    "TEGR1": lambda: tegr1_app.servable(),
                    "TEGR2": lambda: tegr2_app.servable(),
                    "TEGR3": lambda: tegr3_app.servable(),
                },
                admin=admin,
                show=False,
                port=port,
            )


if __name__ == "__main__":
    main()
