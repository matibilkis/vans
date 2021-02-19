amplitude=$1
. ~/qenv_bilkis/bin/activate
cd ~/vans
python3 main.py --Nphases 2 --amplitude $amplitude --Npriors 200 --photodetectors 8
deactivate
