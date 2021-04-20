sudo docker network create faceswap
sudo docker run -p 4222:4222 -p 8222:8222 -p 6222:6222 -d --name nats-server --network faceswap nats:latest
docker run -p 9000:9000 -e "MINIO_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE" -e "MINIO_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" -d --network faceswap --name minio minio/minio server /dat
docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data -d --network faceswap --name neo4j neo4j
