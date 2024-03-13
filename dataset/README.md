# Dataset Creation

This directory contains the tools to create a dataset of PokeMon sprites from the [PokeAPI/sprites](https://github.com/PokeAPI/sprites) repository.

## Useage
```python3 main.py -h```

The script depends on two configuration files (blacklist.json and dataset_config.json by default).

blacklist.json contains the fields: good_directories, and bad_files. Files are loaded from good_directories, and the bad_files are filtered out.

dataset_config.json specifies the source_directory (the path to the PokeAPI/sprites repository) and the output_directory (the path to the dataset to be created). It also contains the option to disable blacklist filtering, as well as the desired dimension of the sprite images.
