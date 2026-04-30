def test_public_entrypoints_import():
    import src.inference_pipeline  # noqa: F401
    import src.pipeline  # noqa: F401
    import src.model.training.train  # noqa: F401
