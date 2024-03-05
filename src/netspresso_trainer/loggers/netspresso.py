import os

import requests
from bson.objectid import ObjectId
from loguru import logger
from pymongo.mongo_client import MongoClient

MONGODB_TEMP_URI = ""


class ModelSearchServerHandler:
    def __init__(self, task, model, mongodb_uri: str=MONGODB_TEMP_URI) -> None:
        client = MongoClient(mongodb_uri)

        try:
            client.admin.command('ping')
            logger.debug("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            raise e

        self._db = client['custom-training-board']['trainer-all-in-one']
        self._session_id = None

        self._create_session(title=f"[{task}]{model}")


    def init_epoch(self):
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = int(value)

    def _is_ready(self):
        return self._session_id is not None

    def _append(self, scalar_dict, mode='train'):
        assert self._is_ready()
        meta_string = f"{mode}/" if mode is not None else ""
        contents = {'$push': {f"{meta_string}{key}": {'epoch': self._epoch, 'value': value}
                              for key, value in scalar_dict.items()},
                    '$currentDate': {'lastModified': True }}
        result = self._db.update_one({'_id': self._session_id}, contents, upsert=True)
        return result

    def _create_session(self, title: str ="test") -> ObjectId:
        example_document = { "title": title }
        document = self._db.insert_one(example_document)
        self._session_id = document.inserted_id
        return self._session_id

    def create_session(self, title: str="test") -> ObjectId:
        return self._create_session(title=title)

    def log_scalar(self, key, value, mode='train'):
        result = self._append({key: value}, mode=mode)
        return result

    def log_scalars_with_dict(self, scalar_dict, mode='train'):
        result = self._append(scalar_dict, mode=mode)
        return result

    def __call__(self,
            train_losses, train_metrics, valid_losses, valid_metrics,
            learning_rate, elapsed_time,
        ) -> None:

        self.log_scalars_with_dict(train_losses, mode='train')
        self.log_scalars_with_dict(train_metrics, mode='train')

        if valid_losses is not None:
            self.log_scalars_with_dict(valid_losses, mode='valid')
        if valid_metrics is not None:
            self.log_scalars_with_dict(valid_metrics, mode='valid')

        if learning_rate is not None:
            self.log_scalar('learning_rate', learning_rate, mode='misc')
        if elapsed_time is not None:
            self.log_scalar('elapsed_time', elapsed_time, mode='misc')
