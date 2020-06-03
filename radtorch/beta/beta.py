
from ..settings import *






def colab_gui():
  !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
  !unzip -qq ngrok-stable-linux-amd64.zip
  !wget -O data.zip http://dl.dropboxusercontent.com/s/ssntnbpu3owsthv/alexmed_data.zip?dl=0 -q
  get_ipython().system_raw('./ngrok http 8501 &')
  ! curl -s http://localhost:4040/api/tunnels | python3 -c \
      "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
