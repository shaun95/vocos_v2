# Vocos v2

My implementation of Vocos([paper](https://arxiv.org/abs/2306.00814)) v2 for JSUT([link](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)).

The difference between original(v1) is
- using ConvNeXt v2 layer


# Requirements

```sh
pip install torch torchaudio lightning
```

or

```sh
docker image build -t vocos -f docker/Dockerfile .
docker container run --rm -it --gpus all -v $(pwd):/work vocos
```


# Usage
Running run.sh will automatically download the data and begin training.  
So just execute the following commands to begin training.

```sh
cd scripts
./run.sh
```

synthesize.sh uses last.ckpt by default, so if you want to use a specific weight, change it.

```sh
cd scripts
./synthesis.sh
```

# Result

WIP
