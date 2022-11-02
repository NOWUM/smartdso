FROM python:3.10-slim
# set time zone to europe berlin
ENV TZ="Europe/Berlin"
# switch to root for install
USER root
# install glpk
RUN apt-get update && apt-get install --no-install-recommends -y gcc g++ libglpk-dev glpk-utils\
   && rm -rf /var/lib/apt/lists/*
# install requirements
COPY ./requirements.txt .
#RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# add user to start python script
RUN useradd -s /bin/bash admin
# make working directory
RUN mkdir /src
RUN chown -R admin /src
# fix pypsa error --> PermissionError: [Errno 13] Permission denied: '/home/admin'
RUN mkdir -p /home/admin
RUN chown -R admin /home/admin
# copy script file to working directory
COPY . /src
# switch to user admin
USER admin
# set working directory
WORKDIR /src
# run script
CMD ["python", "-u" ,"./main.py"]
