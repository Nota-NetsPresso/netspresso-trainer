from netspresso_trainer import trainer


def netspresso_train():
    is_graphmodule_training = False
    trainer(is_graphmodule_training=is_graphmodule_training)
    
def netspresso_train_fx():
    is_graphmodule_training = True
    trainer(is_graphmodule_training=is_graphmodule_training)