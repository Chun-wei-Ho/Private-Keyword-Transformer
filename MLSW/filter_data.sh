# !/bin/bash

set -euxo pipefail

python MLSW/filter_data.py --lang en --word-count-min 4500 --word-count-max 5000
python MLSW/filter_data.py --lang de --word-count-min 4000 --word-count-max 5000
python MLSW/filter_data.py --lang fr --word-count-min 4000 --word-count-max 5000
python MLSW/filter_data.py --lang ru --word-count-min 1000 --word-count-max 5000
