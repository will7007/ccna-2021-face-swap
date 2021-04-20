import asyncio
import os

import minio.error
import numpy as np
from nats.aio.client import Client as NATS
from minio import Minio
from minio.error import S3Error
import pickle
import io
import cv2
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
from nats.aio.utils import new_inbox


async def start(loop):
    await nc.connect("127.0.0.1:4222", loop=loop)


async def stop():
    await nc.close()


async def run(topic, message):
    try:
        response = await nc.request(topic, message.encode("utf-8"), timeout=10)
        return response.data
    except Exception as e:
        print("Error:", e)


def is_entry(name):
    try:
        client_minio.stat_object(buckets["faces"], name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            print("NoSuchKey in get_face")
            return False
        else:
            raise e


def read_image(path):
    with open(path, "rb") as fd:
        return fd.read()


def get_face(name):
    data = get_object(name, "faces")
    if type(data) == bool and not data:
        return False
    else:
        return pickle.loads(get_object(name, "faces"))


def get_object(name, bucket_name):
    try:
        response = client_minio.get_object(buckets[bucket_name], name)
        data = response.data
        response.close()
        response.release_conn()
        return data
    except S3Error as e:
        if e.code == "NoSuchKey":
            print("NoSuchKey in get_face")
            return False  # TODO: return None instead?
        else:
            raise e


def update_face(name, img): set_object(name, "faces", img)


def set_object(name, bucket_name, object_source):
    object_pickled = pickle.dumps(object_source)
    client_minio.put_object(buckets[bucket_name], name, io.BytesIO(object_pickled), len(object_pickled))


def convert_image(img) -> np.ndarray:
    """
    Convert a blob read through fd.read() and turn it into a numpy image.
    :param img: An image retrieved from get_face_sql() or from the disk via .read()ing a file descriptor.
    :return: A numpy version of img.
    """
    return cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)


def create():
    filepath = input("Enter in the file path of the face you would like to store: ")
    while filepath == "" or not os.path.isfile(filepath):
        filepath = input("Invalid file, try again: ")
    name = ""
    while name == "":
        name = input("Enter in a name for the face you just selected: ")
    if is_entry(name):
        print("Sorry, " + name + " is already being used in the database")
    else:
        update_face(name, read_image(filepath))
        print(loop.run_until_complete(run("create", name)))


def list_faces():
    print("Here is a list of faces currently in the database:")
    print(loop.run_until_complete(run("list", "")))


def swap():
    donor = input("Enter in the donor face which will be transplanted to the base face: ")
    while donor == "":  # or not get_face(donor):
        donor = input("This donor face is not present in the database, please try again: ")
    base = input("Enter in the base face which will have its face overwritten by the donor face: ")
    while base == "":  # not get_face(base):
        base = input("This base face is not present in the database, please try again: ")
    swap_name_input = donor + "|" + base
    swap_name = loop.run_until_complete(run("swap", swap_name_input))
    if swap_name is not None and swap_name != "":
        cv2.imshow("Viewing face: " + swap_name_input, get_face(swap_name_input))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rating = input("Was that face swap good? Enter in Y or y if it was or anything else if it wasn't.")
        if rating == "Y" or rating == "y":
            print("Great, glad you liked it!")
            liked = 1
        else:
            print("Oh, sorry about that!")
            liked = 0
        client_minio.remove_object(buckets["faces"], swap_name)
        print(loop.run_until_complete(run("rank", swap_name_input + "|" + str(liked))))
    else:
        print("Error swapping faces")


def delete():
    name = input("Enter in the face which will be removed from the database: ")
    if name == "":  # or not is_entry(name):  # API server won't need to check this
        print("I guess there's nothing you want to delete, never mind")
    else:
        print(loop.run_until_complete(run("delete", name)))


def view():
    name = input("Enter in the face that you want to see: ")
    if name == "":  # or not is_entry(name):  # API server won't need to check this
        print("I guess there's nothing you want to see, never mind")
    else:
        rating = loop.run_until_complete(run("view", name)).decode()
        if rating is not None and rating != "":
            cv2.imshow("Viewing face: " + name + ", ratings are " + rating, convert_image(get_face(name)))
            # unswapped image is not yet of MAT type
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    nc = NATS()
    loop.run_until_complete(start(loop))

    client_minio = Minio(
        "localhost:9000",
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        secure=False
    )
    buckets = {"faces": "face-images", "points": "facial-points"}
    for bucket in buckets.values():
        found = client_minio.bucket_exists(bucket)
        if not found:
            client_minio.make_bucket(bucket)
        else:
            print("Bucket '{}' already exists".format(bucket))

    while True:
        command = input("What command would you like to run? [create, list, swap, delete, view, stop]: ")
        if command == "stop":
            break
        elif command == "create":
            create()
        elif command == "list":
            list_faces()
        elif command == "swap":
            swap()
        elif command == "delete":
            delete()
        elif command == "view":
            view()
        else:
            loop.run_until_complete(run(command, "User says " + command))

    loop.run_until_complete(stop())
    loop.close()
