FROM python:3.12-alpine

RUN adduser -D pbl6

WORKDIR /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apk update
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

RUN chown -R pbl6:pbl6 /app
USER pbl6

EXPOSE 5000

CMD [ "gunicorn", "-w 4", "-t 1800", "-b", "0.0.0.0:5000", "app:app" ]
