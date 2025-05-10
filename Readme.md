The guide below is only tested under windows.

## Build command:
In the `/python` directory, run:

```bash
docker build --build-arg DOCKER_USER=myuser --build-arg DOCKER_UID=1000 --build-arg DOCKER_GID=1000 -t tf_mnist .
```

## Run Command:
To run the container:

```bash
docker run --gpus all -it -p 8888:8888 -p 6006:6006 -v %cd%:/home/myuser/work --name tf_mnist_container tf_mnist
```

- 6006:6006 is a portmapping so we can access the tensorboard from our browser

## Running a pyton file inside of docker:

```
cd work
```

```
python data.py
```

## Open Tensorboard:

```
tensorboard --logdir logs
```

Open under: http://localhost:6006/

## Clear logs from previous runs:

```
rm -rf ./logs/
```