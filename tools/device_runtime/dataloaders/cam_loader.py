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

import cv2

class LoadCamera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)

    def __iter__(self):
        return self

    def __next__(self):
        success, img = self.cap.read()
        if success:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            raise IOError("Failed to read camera frame")

    def __len__(self):
        return 1e12 # Set a large number to make it long enough
