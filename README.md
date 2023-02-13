# Angel System ASR Server

## Create conda environment

```
conda env create -f asr.yml
conda activate angel_system_asr
```

## Running the Server

```
export CUDA_VISIBLE_DEVICES=4; python speech_server.py
```

## Running the Client

Ensure the server is actively running on the server machine. Also ensure the client is
connected to a microphone peripheral. This script will indicate when
recording is active. The ASR-parsed server response should be printed in the console.
```
python speech_client.py
```

