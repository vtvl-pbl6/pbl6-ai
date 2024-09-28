FROM python:3.12-slim

# Install OpenJDK-8
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1

RUN adduser --disabled-password pbl6

WORKDIR /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

RUN chown -R pbl6:pbl6 /app
USER pbl6

EXPOSE 5000

CMD [ "gunicorn", "-w 2", "-t 1800", "-b", "0.0.0.0:5000", "app:app" ]
