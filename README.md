# Angel System ASR Server

## Running the Server

### Create conda environment

```
conda env create -f speech_server.yml
conda activate speech_server
```

The server can then be instantiated with:

```
export CUDA_VISIBLE_DEVICES=4; python speech_server.py
```

## Running the Client

### Create conda environment

```
conda env create -f speech_client.yml
conda activate speech_client
```

Ensure the server is actively running on the server machine.
Also ensure the client is connected to a microphone peripheral.
This script will indicate when recording has begun. Otherwise, you can
optionally pass in a prerecorded file using the `-f/--file` flag.
```
python speech_client.py --asr/--vd
```
