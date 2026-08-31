import torch

from model import FunASRNano


def _model_with_ctc_modules():
    model = object.__new__(FunASRNano)
    torch.nn.Module.__init__(model)
    model.ctc_decoder = torch.nn.Linear(2, 2)
    model.ctc = torch.nn.Linear(2, 2)
    model.ctc_tokenizer = object()
    model.blank_id = 2
    model._externally_loaded_ctc_keys = set()
    return model


def _complete_ctc_keys(model):
    keys = {f"ctc_decoder.{key}" for key in model.ctc_decoder.state_dict()}
    keys.update(f"ctc.{key}" for key in model.ctc.state_dict())
    return keys


def test_checkpoint_without_ctc_weights_disables_timestamp_modules():
    model = _model_with_ctc_modules()

    model.on_pretrained_model_loaded(set())

    assert model.ctc_decoder is None
    assert model.ctc is None
    assert model.ctc_tokenizer is None
    assert model.blank_id is None


def test_checkpoint_with_complete_ctc_weights_keeps_timestamp_modules():
    model = _model_with_ctc_modules()

    model.on_pretrained_model_loaded(_complete_ctc_keys(model))

    assert model.ctc_decoder is not None
    assert model.ctc is not None
    assert model.ctc_tokenizer is not None
    assert model.blank_id == 2


def test_external_ctc_checkpoint_keys_count_toward_completeness():
    model = _model_with_ctc_modules()
    model._externally_loaded_ctc_keys = _complete_ctc_keys(model)

    model.on_pretrained_model_loaded(set())

    assert model.ctc_decoder is not None
    assert model.ctc is not None
