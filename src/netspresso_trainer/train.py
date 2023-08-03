from netspresso_trainer import trainer


def netspresso_train():
    trainer(is_graphmodule_training=False)
    
def netspresso_train_fx():
    trainer(is_graphmodule_training=True)