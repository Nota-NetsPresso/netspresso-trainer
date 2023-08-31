<!-- FIXME: copied from https://github.com/Nota-NetsPresso/PyNetsPresso/blob/main/README.md?plain=1 -->

### Automatic Compression

Automatically compress the model by setting the compression ratio for the model.

Enter the ID of the uploaded model, the name and storage path of the compressed model, and the compression ratio.

```python
compressed_model = compressor.automatic_compression(
    model_id=model.model_id,
    model_name="YOUR_COMPRESSED_MODEL_NAME",
    output_path="OUTPUT_PATH",  # ex) ./compressed_model.h5
    compression_ratio=0.5,
)
```

