# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import torch
import torch.fx as fx
import torch.nn as nn

__all__ = ['convert_graphmodule', 'save_graphmodule']

def _convert_graphmodule(model: nn.Module) -> fx.GraphModule:
    try:
        _graph = fx.Tracer().trace(model)
        model = fx.GraphModule(model, _graph)
        return model
    except Exception as e:
        raise e


def convert_graphmodule(model: nn.Module) -> fx.GraphModule:
    return _convert_graphmodule(model)


def save_graphmodule(model: nn.Module, f):
    model: fx.GraphModule = _convert_graphmodule(model)
    torch.save(model, f)
