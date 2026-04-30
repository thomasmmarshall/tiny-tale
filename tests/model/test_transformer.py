import torch

from src.model.architecture.transformer import FeedForward, RMSNorm, Transformer, TransformerConfig


def test_transformer_forward_and_loss():
    config = TransformerConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=8,
        num_key_value_heads=2,
    )
    model = Transformer(config)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert logits.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(loss)


def test_attention_is_causal():
    config = TransformerConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        use_rope=False,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )
    model = Transformer(config).eval()
    prefix = torch.tensor([[1, 2, 3]])
    extended = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        prefix_logits = model(prefix)
        extended_logits = model(extended)

    assert torch.allclose(prefix_logits[:, :3], extended_logits[:, :3], atol=1e-5)


def test_modern_block_components_are_used():
    config = TransformerConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        use_rope=True,
    )
    model = Transformer(config)

    assert isinstance(model.layers[0].input_norm, RMSNorm)
    assert isinstance(model.layers[0].feed_forward, FeedForward)
    assert model.position_embeddings is None
    assert model.layers[0].attention.num_key_value_heads == 2


def test_invalid_grouped_query_config_rejected():
    try:
        TransformerConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=3,
            intermediate_size=32,
            max_position_embeddings=8,
        )
    except ValueError as exc:
        assert "num_attention_heads" in str(exc)
    else:
        raise AssertionError("Expected invalid GQA config to fail")
