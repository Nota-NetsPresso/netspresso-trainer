<!-- FIXME: copied from https://github.com/Nota-NetsPresso/PyNetsPresso/blob/main/README.md?plain=1 -->

Use **[PyNetsPresso]** for a seamless model optimization process.  

PyNetsPresso resolves AI-related constraints in business use cases and enables cost-efficiency and enhanced performance by removing the requirement for high-spec servers and network connectivity and preventing high latency and personal data breaches.

**PyNetsPresso** is a python interface with the NetsPresso web application and REST API.

Easily compress various models with our resources. Please browse the [Docs] for details, and join our [Discussion Forum] for providing feedback or sharing your use cases.

To get started with the PyNetsPresso, you will need to sign up either at [NetsPresso] or PyNetsPresso.</a>

</br>
</br>


## Installation

There are two ways you can install the PyNetsPresso: **1) using pip** or **2) manually through our project GitHub repository**.

To install this package, please use **Python 3.8** or higher.

From PyPI (Recommended)
```bash
pip install netspresso
```

From GitHub
```bash
git clone https://github.com/nota-netspresso/pynetspresso.git
cd pynetspresso
pip install -e .
```


## Quick Start

### Login

To use the PyNetsPresso, please enter the email and password registered in [NetsPresso].

```python
from netspresso.client import SessionClient
from netspresso.compressor import ModelCompressor

session = SessionClient(email='YOUR_EMAIL', password='YOUR_PASSWORD')
compressor = ModelCompressor(user_session=session)
```

### Upload Model

To upload your trained model, simply enter the required information. 

When a model is successfully uploaded, a unique `model.model_id` is generated to allow repeated use of the uploaded model.

```python
from netspresso.compressor import Task, Framework

model = compressor.upload_model(
    model_name="YOUR_MODEL_NAME",
    task=Task.IMAGE_CLASSIFICATION,
    framework=Framework.TENSORFLOW_KERAS,
    file_path="YOUR_MODEL_PATH", # ex) ./model.h5
    input_shapes="YOUR_MODEL_INPUT_SHAPES",  # ex) [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
)
```

</br>
</br>



[Docs]: https://nota-netspresso.github.io/PyNetsPresso-docs
[Discussion Forum]: https://github.com/orgs/Nota-NetsPresso/discussions
[NetsPresso]: https://netspresso.ai?utm_source=git_comp&utm_medium=text_np&utm_campaign=py_launch
[PyNetsPresso]: https://py.netspresso.ai/?utm_source=git_comp&utm_medium=text_py&utm_campaign=py_launch
[here]: https://www.nota.ai/contact-us
[contact@nota.ai]: mailto:contact@nota.ai