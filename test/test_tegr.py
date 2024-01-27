def test_tegr1():
    from tqf.tegr.tegr1 import tegr1_qf
    results = tegr1_qf.view_results(tabulator=False, to_csv=True)
    print("\nTEGR1 Results")
    print(results)


def test_tegr2():
    from tqf.tegr.tegr2 import tegr2_qf
    results = tegr2_qf.view_results(tabulator=False, to_csv=True)
    print("\nTEGR2 Results")
    print(results)


def test_tegr3():
    from tqf.tegr.tegr3 import tegr3_qf
    results = tegr3_qf.view_results(tabulator=False, to_csv=True)
    print("\nTEGR3 Results")
    print(results)
