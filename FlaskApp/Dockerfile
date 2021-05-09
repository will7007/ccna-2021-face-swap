FROM opencvcourses/opencv-docker AS build
# we will freeze our Python script later

WORKDIR /src
RUN apt update
COPY requirements.txt /src/
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install neo4j minio asyncio-nats-client
RUN python3 -m pip install pillow
#RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install flask_bootstrap wtforms werkzeug FlaskApp
COPY app.py util.py /src/
COPY static /src/static/
COPY templates /src/templates/
# RUN python3 -m pip install pyinstaller
# RUN pyinstaller app.py
#CMD ["ApiServer.py"]
#ENTRYPOINT ["python3.7"]
CMD ["flask","run","--host=0.0.0.0"]
#ENTRYPOINT ["flask run"]

# FROM debian:buster-slim
# WORKDIR /bin
# RUN apt update && apt upgrade -y
# RUN apt install python3.7 -y
# COPY static /bin/static/
# COPY templates /bin/templates/
# COPY --from=build /src/dist/ApiServer/ApiServer /bin/ApiServer
# ENTRYPOINT ["/bin/ApiServer"]
