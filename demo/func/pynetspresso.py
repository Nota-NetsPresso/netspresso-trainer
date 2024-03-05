from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import shutil

import gradio as gr
from netspresso import NetsPresso

class InputShapes(TypedDict):
    batch: int
    channel: int
    dimension: List[int]  # Height, Width

class NetsPressoSession:
    
    def __init__(self) -> None:
        self.compressor = None
        self._is_verified = False

    @property
    def is_verified(self) -> bool:
        return self._is_verified

    def login(self, email: str, password: str) -> bool:
        try:
            netspresso = NetsPresso(email=email, password=password)
            self.compressor = netspresso.compressor()
            self._is_verified = True
        except Exception as e:
            self._is_verified = False
            raise e

    def compress(self, model_name: str, model_path: Union[Path, str],
                 batch_size: int, channels: int, height: int, width: int,
                 compression_ratio: float,
                 compressed_model_dir: Union[Path, str]) -> Path:

        if not self._is_verified:
            raise gr.Error("Please log in first at the console on the left side.")

        if self.compressor is None:
            self._is_verified = False
            raise gr.Error("The session is expired! Please log in again.")

        model_path = Path(model_path)
        if not model_path.exists():
            raise gr.Error(f"Model path {str(model_path)} not found!")

        if compressed_model_dir is None or compressed_model_dir == "":
            compressed_model_dir = model_path.with_name(f"{model_path.stem}_compressed")
        compressed_model_dir = Path(compressed_model_dir)
        
        if compressed_model_dir.exists():
            shutil.rmtree(compressed_model_dir)

        if model_name is None or model_name == "":
            model_name = model_path.stem
            
        compression_result = self.compressor.automatic_compression(
            input_shapes=[InputShapes(batch=batch_size, channel=channels, dimension=[height, width])],
            input_model_path=str(model_path),
            output_dir=str(compressed_model_dir),
            compression_ratio=compression_ratio,
        )

        if compression_result['status'] != 'completed':
            status_info = compression_result['status']
            raise RuntimeError(f"Failed at NP compressor! Status info: {status_info}")    
        return compression_result['compressed_model_path']
        


def login_with_session(session: NetsPressoSession, email: str, password: str) -> NetsPressoSession:
    try:
        session.login(email, password)
        if session.is_verified:
            gr.Info("Login success!")
            return session
    except Exception as e:
        raise gr.Error(
            f"We're sorry, but login failed with an error: {str(e)}"
        ) from e


def compress_with_session(
    session: NetsPressoSession,
    model_name: str, model_path: Union[Path, str],
    batch_size: int, channels: int, height: int, width: int,
    compression_ratio: float,
    compressed_model_dir: Optional[Union[Path, str]]
) -> List[Union[NetsPressoSession, str]]:
    try:
        output_path = session.compress(
            model_name=model_name,
            model_path=model_path,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            compression_ratio=compression_ratio,
            compressed_model_dir=compressed_model_dir
        )
        gr.Info("Compress success!")
        return [session, output_path]
    except Exception as e:
        output_path = "Error!"
        raise gr.Error(
            f"Error while compressing the model with NetsPresso: {str(e)}"
        ) from e
