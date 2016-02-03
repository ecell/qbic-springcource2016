FROM ubuntu:14.04

RUN apt-get update; apt-get install -y python-dev pandoc python python-pip python-zmq; pip install jupyter
