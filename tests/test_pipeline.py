from src.pipeline import Pipeline


def test_pipeline_data_module_uses_processed_paths(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    raw_train = tmp_path / "train.txt"
    raw_val = tmp_path / "valid.txt"
    raw_train.write_text("Hello <b>world</b>.\n", encoding="utf-8")
    raw_val.write_text("Validation <i>text</i>.\n", encoding="utf-8")
    config_path.write_text(
        f"""
data:
  train_path: "{raw_train}"
  val_path: "{raw_val}"
  cleaning:
    remove_html: true
    remove_special_chars: true
    lowercase: true
    min_length: 1
    max_length: 1000
  dataloader:
    batch_size: 2
    max_length: 8
    num_workers: 0
    shuffle_buffer_size: 4
tokenizer:
  min_frequency: 1
  special_tokens:
    <pad>: "[PAD]"
    <unk>: "[UNK]"
    <bos>: "[BOS]"
    <eos>: "[EOS]"
model:
  vocab_size: 32
  hidden_size: 8
  num_hidden_layers: 1
  num_attention_heads: 2
  intermediate_size: 16
  max_position_embeddings: 8
training:
  learning_rate: 1.0e-3
  max_steps: 1
logging:
  wandb_project: "test"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    pipeline = Pipeline(str(config_path), "exp")
    processed = pipeline.process_data()

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, **kwargs):
            return [1, 2, 3]

    data_module = pipeline.setup_data_module(Tokenizer(), processed["paths"])

    assert processed["train_texts"] == ["hello world ."]
    assert data_module.train_path == processed["paths"]["train"]
    assert data_module.val_path == processed["paths"]["val"]
