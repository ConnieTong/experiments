python tf_lstm_train.py --verbose=10 --model=lstm --seq_len=25 --nb_lstm_layers=1 --weights=0 --use_linear_interpolate=1 &> out.txt
python tf_lstm_train.py --verbose=10 --model=lstm --seq_len=50 --nb_lstm_layers=1 --weights=0 --use_linear_interpolate=1 &>> out.txt
python tf_lstm_train.py --verbose=10 --model=lstm --seq_len=100 --nb_lstm_layers=1 --weights=0 --use_linear_interpolate=1 &>> out.txt
python tf_lstm_train.py --verbose=10 --model=lstm --seq_len=25 --nb_lstm_layers=1 --weights=1 --use_linear_interpolate=1 &>> out.txt
python tf_lstm_train.py --verbose=10 --model=lstm --seq_len=50 --nb_lstm_layers=1 --weights=1 --use_linear_interpolate=1 &>> out.txt
python tf_lstm_train.py --verbose=10 --model=lstm --seq_len=100 --nb_lstm_layers=1 --weights=1 --use_linear_interpolate=1 &>> out.txt
