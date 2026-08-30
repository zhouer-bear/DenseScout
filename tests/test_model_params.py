from densescout.model import DenseScout, count_parameters


def test_model_parameter_count_about_one_million():
    n = count_parameters(DenseScout(pretrained=False))
    assert 900_000 <= n <= 1_200_000
