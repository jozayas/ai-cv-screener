from cv_screener import main


def test_main_prints_placeholder_message(capsys) -> None:
    main()

    assert capsys.readouterr().out == "Hello from cv-screener!\n"
