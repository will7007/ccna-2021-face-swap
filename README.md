# CCNA 2021: Face-Swapping Cloud Native Application
A distributed application that will (eventually) allow a user to perform CRUD operations on a facial database (which will be full of faces that users uploaded) and ask the server to place the face of one person onto another person's head.
The server will be able to pre-compute and store the coordinates of the facial features for each image, so that the only computation required when a facial swap is performed will be the warping of triangles and smooth-copying of the new face onto the old head.

# List of software components
## Application Component
While Python and Go microservices would ideally make up our application, a monolithic application can be used to meet the requirements of the project for the sake of time.
## Datastore Component
Neo4j is being used to store metadata about the faces (especially the directed edges between faces which serve to rate face swap parings) whlie MinIO is being used to hold the image blobs and facial coordinates.
## API Server
Go will be used to create the API server, with NATS being used to communicate between the API server and microservices.
## Client
The REST client will be made in Python, but we are open to developing a mobile/web client later if we have time.

# Milestones
Note that the dates are incorrect due to the first milestone being delayed.
> Milestone 1: (Deadline: March 25) (DONE)
> Develop the application and external client to test the application. Any storage aspect is done locally. At this stage client is the test program. You can use any programming language depending on the application. 
>                                                                             
> Milestone 2: (Deadline: April 1) (DONE)
> Move the storage to a datastore (run as a container). Integrate application and storage.
>                                                                             
> Milestone 3: (Deadline: April 8)                                            
> Run NATS messaging system as a Docker container. Familiarize yourself with publishing and subscribing the messages that your application consume/produces from/to NATS
>                                                                             
> Milestone 4: (Deadline: April 8)                                            
> Integrate your application and external client with NATS. At this point you have
> External client -> NATS -> Application                                      
> External client <- NATS <- Application                                      
>                                                                             
> External client publishes input, which the application subscribes           
> Application publishes output, which the External client consumes, and displays
>                                                                             
> Milestone 5: (Deadline: April 15)                                           
> Containerize the application. At this point, application, NATS, and datastore all run as containers.
>                                                                             
> Milestone 6: (Deadline: April 15)                                           
> Design the REST API for the application for the outside world. Develop the APIServer to which the external client connects. The external client is a user of the API. The APIServer publishes messages to NATS, and subscribes messages from NATS. 
>                                                                             
> You can mock your application for this milestone.                           
>                                                                             
> Milestone 7: (Deadline: April 22)                                           
> Containerize the APISserver. At this point, all components, with the exception of external client run as containers.
>                                                                             
> Milestone 8: (Deadline: April 22)                                           
> Integrate all components (External client, APIServer, NATS, application, datastore), performing end-to-end testing
>                               
