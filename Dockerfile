FROM ubuntu:14.04

RUN apt-get update; apt-get install -y pandoc python python-dev python-opencv python-numpy python-scipy python-pip python-zmq libfreetype6-dev libpng-dev pkg-config; pip install jupyter matplotlib pip install chainer
