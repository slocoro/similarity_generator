### Similarity generator

This repository contains code to generate similarity scores.

To generate similarities follow the steps below:
- add data set (csv) to data/ folder in root
- run "python src/main.py {filename} {index_column}" from terminal (e.g. "python src/main.py sample_data.csv id")

{filename} -> name of file to be processed
{index_column} -> name of id column

By default, cosine similarity and all label columns are used to generate similarities.

Similarities are saved in output/ folder in root.