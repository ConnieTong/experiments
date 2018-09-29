#!/bin/bash -l
#SBATCH --gres=gpu:4 -t 5:00:00 --mem=10G -c 3
#SBATCH --constraint='kepler|pascal|volta'

module load anaconda2

python -u train_with_args.py --limit_size -1 --pred_win_size 1 --nb_epc 20 --nb_lstm_layers 1 --model lstm
python -u train_with_args.py --limit_size -1 --pred_win_size 2 --nb_epc 20 --nb_lstm_layers 1 --model lstm
python -u train_with_args.py --limit_size -1 --pred_win_size 3 --nb_epc 20 --nb_lstm_layers 1 --model lstm

python -u train_with_args.py --limit_size -1 --pred_win_size 1 --nb_epc 20 --nb_lstm_layers 1 --model deep_lstm
python -u train_with_args.py --limit_size -1 --pred_win_size 1 --nb_epc 20 --nb_lstm_layers 2 --model deep_lstm
python -u train_with_args.py --limit_size -1 --pred_win_size 1 --nb_epc 20 --nb_lstm_layers 3 --model deep_lstm
