"""
    default configuration of the network
    change them to better fit your model
"""


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


cl = 40
scnt = 91
pcnt = 19

args = dotdict({
    'lr': 1e-5,
    'l2': 0.4,
    'dropout': 0.3,
    'epochs': 800,
    'batch_size': 4,
})
