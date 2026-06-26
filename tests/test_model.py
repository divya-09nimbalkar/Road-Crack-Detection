from src import model

def test_model_build():
    net = model.build_model()
    assert net is not None
