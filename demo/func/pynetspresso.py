from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, TypedDict

import gradio as gr

from netspresso.client import SessionClient
from netspresso.compressor import ModelCompressor, Task, Framework


class InputShapes(TypedDict):
    batch: int
    channel: int
    dimension: List[int]  # Height, Width


class NetsPressoSession:
    task_dict = {
        'classification': Task.IMAGE_CLASSIFICATION,
        'segmentation': Task.SEMANTIC_SEGMENTATION
    }

    def __init__(self) -> None:
        self.compressor = None
        self._is_verified = False

    @property
    def is_verified(self) -> bool:
        return self._is_verified

    def login(self, email: str, password: str) -> bool:
        try:
            session = SessionClient(email=email, password=password)
            self.compressor = ModelCompressor(user_session=session)
            self._is_verified = True
        except Exception as e:
            self._is_verified = False
            raise e

    def compress(self, model_name: str, task: str, model_path: Union[Path, str],
                 batch_size: int, channels: int, height: int, width: int,
                 compression_ratio: float,
                 compressed_model_path: Union[Path, str]) -> Path:

        if not self._is_verified:
            raise gr.Error(f"Please log in first at the console on the left side.")

        if self.compressor is None:
            self._is_verified = False
            raise gr.Error(f"The session is expired! Please log in again.")

        if task not in self.task_dict:
            raise gr.Error(f"Selected task is not supported in web UI version.")

        model_path = Path(model_path)
        if not model_path.exists():
            raise gr.Error(f"Model path {str(model_path)} not found!")

        if compressed_model_path is None or compressed_model_path == "":
            compressed_model_path = model_path.with_name(f"{model_path.stem}_compressed{model_path.suffix}")

        if model_name is None or model_name == "":
            model_name = model_path.stem

        model = self.compressor.upload_model(
            model_name=model_name,
            task=self.task_dict[task],
            # file_path: e.g. ./model.pt
            file_path=str(model_path),
            # input_shapes: e.g. [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
            input_shapes=[InputShapes(batch=batch_size, channel=channels, dimension=[height, width])],
            framework=Framework.PYTORCH
        )

        _ = self.compressor.automatic_compression(
            model_id=model.model_id,
            model_name=model_name,
            # output_path: e.g. ./compressed_model.pt
            output_path=str(compressed_model_path),
            compression_ratio=compression_ratio,
        )

        return Path(compressed_model_path)


def login_with_session(session: NetsPressoSession, email: str, password: str) -> NetsPressoSession:
    try:
        session.login(email, password)
        if session.is_verified:
            gr.Info("Login success!")
            return session
    except Exception as e:
        raise gr.Error(
            f"We're sorry, but login failed with an error: {str(e)}"
        )


def compress_with_session(
    session: NetsPressoSession,
    model_name: str, task: Task, model_path: Union[Path, str],
    batch_size: int, channels: int, height: int, width: int,
    compression_ratio: float,
    compressed_model_path: Optional[Union[Path, str]]
) -> List[Union[NetsPressoSession, str]]:
    try:
        output_path = session.compress(
            model_name=model_name, task=task, model_path=model_path,
            batch_size=batch_size, channels=channels, height=height, width=width,
            compression_ratio=compression_ratio,
            compressed_model_path=compressed_model_path
        )
        gr.Info("Compress success!")
        return [session, output_path]
    except Exception as e:
        output_path = "Error!"
        raise gr.Error(
            f"Error while compressing the model with NetsPresso: {str(e)}"
        )
