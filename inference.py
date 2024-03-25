from netspresso_trainer import inference_cli


if __name__ == '__main__':
    logging_dir = inference_cli()

    # With inline yaml
    # logging_dir = evaluation_with_inline_yaml()
    
    print(f"Inference results are saved at: {logging_dir}")