def test_tegr1():
    from tqf.tegr.tegr1 import qf

    results = qf.view_results(tabulator=False, to_csv=True)
    print("\nTEGR1 Results")
    print(results)


def test_tegr2():
    from tqf.tegr.tegr2 import qf

    results = qf.view_results(tabulator=False, to_csv=True)
    print("\nTEGR2 Results")
    print(results)


def test_tegr3():
    from tqf.tegr.tegr3 import qf

    results = qf.view_results(tabulator=False, to_csv=True)
    print("\nTEGR3 Results")
    print(results)
