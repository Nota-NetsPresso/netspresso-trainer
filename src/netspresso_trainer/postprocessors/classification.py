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

from typing import Optional

from ..models.utils import ModelOutput



class ClassificationPostprocessor():
    def __init__(self, conf_model):
        params = conf_model.postprocessor.params
        self.topk_max = params.topk_max

    def __call__(self, outputs: ModelOutput, k: Optional[int]=None):
        pred = outputs['pred']
        maxk = min(self.topk_max, pred.size()[1])
        if k:
            maxk = min(k, maxk)
        _, pred = pred.topk(maxk, 1, True, True)
        return pred.detach().cpu().numpy()
