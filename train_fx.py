from netspresso_trainer import set_arguments, train

if __name__ == '__main__':
    args_parsed, args = set_arguments(is_graphmodule_training=True)
    train(args_parsed, args, is_graphmodule_training=True)