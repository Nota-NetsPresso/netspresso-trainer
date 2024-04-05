from netspresso_trainer import evaluation_cli


if __name__ == '__main__':
    logging_dir = evaluation_cli()

    # With inline yaml
    # logging_dir = evaluation_with_inline_yaml()
    
    print(f"Evaluation results are saved at: {logging_dir}")