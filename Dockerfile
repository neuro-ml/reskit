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
	zlib1g-dev \
  zip

RUN pip3 install -U pip
RUN pip3 install -U notebook

ADD . reskit

RUN pip3 install -r reskit/requirements.txt
RUN zip -r reskit.zip reskit
RUN pip3 install -U reskit.zip
