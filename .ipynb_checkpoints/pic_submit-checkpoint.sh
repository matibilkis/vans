mplitude=$1
. ~/qenv_bilkis/bin/activate
cd ~/unfolding_qreceiver
python3 main.py --Nphases 2 --amplitude $amplitude --Npriors 200 --photodetectors 8
deactivate

