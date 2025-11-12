from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam

def optim_entry(config):
    return globals()[config['type']](**config['kwargs'])