import os

from datasets.utils.parsers.parser_image_folder import ParserImageFolder


def create_parser(name, root, split='train', **kwargs):
    name = name.lower()
    name = name.split('/', 2)
    prefix = ''
    if len(name) > 1:
        prefix = name[0]
    name = name[-1]

    assert os.path.exists(root)
    parser = ParserImageFolder(root, **kwargs)
    return parser
