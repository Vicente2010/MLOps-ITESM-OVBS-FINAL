FROM python:3.10

WORKDIR / 
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt 
RUN apt-get update && apt-get install -y vim 

EXPOSE 8000