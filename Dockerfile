FROM opencvcourses/opencv-docker AS build
# we will freeze our Python script later

WORKDIR /src
RUN apt update
RUN apt install -y cmake
RUN pip3 install dlib
RUN pip3 install numpy
#RUN pip3 install pickle already installed by default
RUN pip3 install neo4j
RUN pip3 install minio
RUN pip3 install asyncio-nats-client
RUN pip3 install werkzeug
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY faceSwap.py shape_predictor_68_face_landmarks.dat /src/
CMD ["faceSwap.py"]
ENTRYPOINT ["python3.7"]
