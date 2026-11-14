"""AI in production (AI & validation).

Only inference lives here, with onnxruntime. Training is in training/ and is
never run by the pipeline, so GitHub Actions does not install PyTorch.
"""
