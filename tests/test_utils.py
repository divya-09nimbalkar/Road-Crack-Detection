from src import utils

def test_log_message(capsys):
    utils.log_message("Hello")
    captured = capsys.readouterr()
    assert "Hello" in captured.out
