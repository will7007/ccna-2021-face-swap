FROM opencvcourses/opencv-docker AS build
# we will freeze our Python script later

WORKDIR /src
COPY ApiServer.py requirements.txt util.py /src/
COPY static /src/static/
COPY templates /src/templates/
RUN apt update
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install neo4j minio asyncio-nats-client
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install flask_bootstrap wtforms werkzeug FlaskApp
# RUN python3 -m pip install pyinstaller
# RUN pyinstaller ApiServer.py
CMD ["ApiServer.py"]
ENTRYPOINT ["python3.7"]

# FROM debian:buster-slim
# WORKDIR /bin
# RUN apt update && apt upgrade -y
# RUN apt install python3.7 -y
# COPY static /bin/static/
# COPY templates /bin/templates/
# COPY --from=build /src/dist/ApiServer/ApiServer /bin/ApiServer
# ENTRYPOINT ["/bin/ApiServer"]
