ARG PHP_VERSION=8.3

FROM php:${PHP_VERSION}-cli

ENV DEBIAN_FRONTEND=noninteractive
ENV ONNXRUNTIME_VERSION=1.22.0

RUN apt-get update

RUN apt-get install -y build-essential
RUN apt-get install -y re2c
RUN apt-get install -y autoconf
RUN apt-get install -y pkg-config
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y libfreetype-dev
RUN apt-get install -y wget

RUN docker-php-ext-configure gd --with-freetype --with-jpeg \
    && docker-php-ext-install -j$(nproc) gd

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz -O /tmp/onnxruntime.tgz \
    && tar -C /usr/local --strip-components=1 -xzf /tmp/onnxruntime.tgz \
    && rm /tmp/onnxruntime.tgz \
    && ldconfig

WORKDIR /app

COPY . .

RUN phpize \
    && ./configure --with-ort-onnx \
    && make -j$(nproc) \
    && make install

RUN echo "extension=ort.so" > /usr/local/etc/php/conf.d/ort.ini

CMD ["/bin/bash"]
