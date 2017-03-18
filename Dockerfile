FROM ubuntu:16.04

MAINTAINER Alexander Ivanov alexander.radievich@gmail.com

RUN apt-get update && apt-get install
RUN apt-get install -yqq \
	python3 \
	python3-pip \
	python3-dev \
	build-essential \
	python3-setuptools \
	libxslt1-dev \
	zlib1g-dev

RUN pip3 install -U pip
RUN pip3 install -U notebook==4.2.3
RUN pip3 install -r https://github.com/neuro-ml/reskit/blob/master/requirements.txt
RUN pip3 install -U https://github.com/neuro-ml/reskit/archive/master.zip
