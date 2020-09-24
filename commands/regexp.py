import json
from datetime import datetime
from pathlib import Path

default_dir = Path()


class DB:
    def __init__(self, dir):
        name = 'aedb.json'
        self.dir = dir / name
        self.data = self._load_all()

    def _save_all(self):
        print(self.data)
        json.dump(
            self.data,
            open(
                self.dir,
                'w'
            )
        )

    def _load_all(self):
        try:
            data = json.load(
                open(
                    self.dir,
                    'r'
                )
            )
        except FileNotFoundError:
            data = {}
            json.dump(
                data,
                open(
                    self.dir,
                    'w'
                )
            )
        return data

    def save(self, experiment):
        try:
            _ = self.data[experiment.__hash__()]
            r = input('warning: experiment {} already exists. Override? (y/N)')
            if r != 'y':
                print('info: ignoring')
                return None
        except KeyError:
            pass
        print('info: saving experiment record...')
        self.data[experiment.__hash__()] = experiment.serialize()
        self._save_all()
        return self.data[experiment.__hash__()]

    def list(self):
        for exp in self.data.values():
            print(exp)


class Experiment:
    def __init__(self, name, model_name, train_loss, val_loss, extra, db):
        self.n = name
        self.m = model_name
        self.tl = float(train_loss)
        self.vl = float(val_loss)
        self.e = extra
        self.db = db

    def save(self):
        c = 0
        while not self.db.save(self) and c < 20:
            print('info: error saving. retrying')
            c += 1

    def __hash__(self):
        return self.m + "_" + self.n

    def __eq__(self, other):
        return other.__hash__() == self.__hash__()

    def serialize(self):
        return {'name': self.n,
                'model': self.m,
                'train_loss': self.tl,
                'val_loss': self.vl,
                'extra': self.e,
                'modificated_on': str(datetime.now())
                }


if __name__ == '__main__':
    model_name = input('query: enter model name: ')
    experiment_name = input('query: enter experiment name: ')
    train_loss = input('query: enter train loss: ')
    val_loss = input('query: enter validation loss: ')
    extra = input('query: add any details (free text): ')
    db = DB(default_dir)
    exp = Experiment(experiment_name, model_name, train_loss, val_loss, extra, db)
    exp.save()
