# app(front-end)
FROM python:3.10

WORKDIR /
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y vim

EXPOSE 3000
CMD ["uvicorn", "frontend_main:app", "--host", "0.0.0.0", "--port", "3000" , "--reload"]