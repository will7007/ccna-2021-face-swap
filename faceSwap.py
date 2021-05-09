#! /usr/bin/env python
# This code is my combination of https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
# (some old face swap code which used hard-coded points)
# and http://dlib.net/face_alignment.py.html (dlib face allignment example, so we can get facial points for any image)

import sys
import numpy as np
import cv2
import dlib
import os
import pickle
import io
import signal
from minio import Minio
from minio.error import S3Error
from neo4j import GraphDatabase
from nats.aio.client import Client as NATS
import asyncio
import werkzeug

# k8s addresses
NEO4J_ADDRESS = "bolt://neo4j-service:7687"
MINIO_ADDRESS = "minio-service:9000"
NATS_ADDRESS = "nats://nats-service:4222"
# normal addresses
# NEO4J_ADDRESS = "bolt://localhost:7687"
# MINIO_ADDRESS = "localhost:9000"
# NATS_ADDRESS = "nats://127.0.0.1:4222"
# normal app, k8s dbs
# NEO4J_ADDRESS = "bolt://192.168.1.79:31620"
# MINIO_ADDRESS = "192.168.1.79:31199"
# NATS_ADDRESS = "nats://192.168.1.79:31348"
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)  # MUST be a 2-element tuple to not throw here

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def swapfacefiles(filename1, filename2):
    donor = cv2.imread(filename1);
    base = cv2.imread(filename2);
    return swapfacenumpy(donor, base)


def precalculate_face(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        return False
    face = sp(img, dets[0])
    points = []
    for i in range(0, 68):
        point = (face.part(i).x, face.part(i).y)
        points.append(point)
    return points
    # There seem to be length problems when finding convex hull and DT since one pre-calculation might find more
    # points on the hull than another
    # # Find convex hull
    # hull = []
    # hullIndex = cv2.convexHull(np.array(points), returnPoints=False)
    # for i in range(0, len(hullIndex)):
    #     # marks points from the original set of points that are in the hull
    #     hull.append(points[int(hullIndex[i])])
    #
    # # Find Delaunay triangulation for convex hull points
    # sizeImg = img.shape
    # rect = (0, 0, sizeImg[1], sizeImg[0])
    # dt = calculateDelaunayTriangles(rect, hull)
    # if len(dt) == 0:
    #     return False, False
    # else:
    #     return hull, dt


def swap_face_calc(img1, points1, img2, points2):
    # Find convex hull
    hull1 = []
    hull2 = []
    # This only calculates the convex hull for the second face and assumes points will be shared across faces
    # We should be able to split this no problem, but do we even need to calculate this every time? Can it stay the same
    # across all faces?
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    for i in range(0, len(hullIndex)):
        # marks points from the original set of points that are in the hull
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # Find Delaunay triangulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = calculateDelaunayTriangles(rect, hull2)
    if len(dt) == 0:
        quit()

    img1Warped = np.copy(img2)
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            # Once again, the sample code calculated the dt but only for one face, and assumed that the triangle points
            # would be shared across faces
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img1, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly
    swapped = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    return cv2.resize(swapped, (512, 512))


def swapfacenumpy(img1, img2):
    dets1 = detector(img1, 1)
    dets2 = detector(img2, 1)
    face1 = sp(img1, dets1[0])
    face2 = sp(img2, dets2[0])

    points1 = []
    points2 = []
    for i in range(0, 68):
        point = (face1.part(i).x, face1.part(i).y)
        points1.append(point)
    for i in range(0, 68):
        point = (face2.part(i).x, face2.part(i).y)
        points2.append(point)

    # Find convex hull
    hull1 = []
    hull2 = []
    # This only calculates the convex hull for the second face and assumes points will be shared across faces
    # We should be able to split this no problem, but do we even need to calculate this every time? Can it stay the same
    # across all faces?
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    for i in range(0, len(hullIndex)):
        # marks points from the original set of points that are in the hull
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # Find Delaunay triangulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = calculateDelaunayTriangles(rect, hull2)
    if len(dt) == 0:
        quit()

    img1Warped = np.copy(img2)
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            # Once again, the sample code calculated the dt but only for one face, and assumed that the triangle points
            # would be shared across faces
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img1, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    # Clone seamlessly
    swapped = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    return cv2.resize(swapped, (512, 512))


def demo_prompt():
    print("\n\nCCNA Project Group 1: Face Swapping")
    print("List of possible operations:")
    print("1: Create face in database")
    print("2: List faces in database")
    print("3: Take a face in the database and combine it with a head in the database")
    print("4: Delete a face from the database")
    print("5: View a face from the database")
    # print("6: Rename a face from the database")
    # print("7: Change the face assigned to a name in the database")
    # print("8: Y-axis flip a face for improved results")
    operation = input("Enter in an operation number [1-5]")
    if operation == "1":
        filepath = input("Enter in the file path of the face you would like to store: ")
        while filepath == "" or not os.path.isfile(filepath):
            filepath = input("Invalid file, try again: ")
        name = ""
        while name == "":
            name = input("Enter in a name for the face you just selected: ")
        if get_face(name):
            print("Sorry, " + name + " is already being used in the database")
        else:
            img = read_image(filepath)  # lena.jpg takes 768.16 KB as a cv2 image vs 28.67 KB as a file to store
            points = precalculate_face(convert_image(img))
            if not points:
                print("No face detected in this image")
            else:
                create_entry(name, img, points)
                print("Face added!")
    elif operation == "2":
        print("Here is a list of faces currently in the database:")
        print(get_entry_list())
    elif operation == "3":
        donor = input("Enter in the donor face which will be transplanted to the base face: ")
        while donor == "" or not get_face(donor):
            donor = input("This donor face is not present in the database, please try again: ")
        base = input("Enter in the base face which will have its face overwritten by the donor face: ")
        while not get_face(base):
            base = input("This base face is not present in the database, please try again: ")
        cv2.imshow("Face Swapped", swap_face_calc(convert_image(get_face(donor)), get_points(donor),
                                                  convert_image(get_face(base)), get_points(base)))
        print("Resulting person has been displayed in a window!")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rating = input("Was that face swap good? Enter in Y or y if it was or anything else if it wasn't.")
        if rating == "Y" or rating == "y":
            print("Great, glad you liked it! This face swap is now rated", update_rating(donor, base, True))
        else:
            print("Oh, sorry about that! This face swap is now rated", update_rating(donor, base, False))
    elif operation == "4":
        name = input("Enter in the face which will be removed from the database: ")
        if name == "" or not get_face(name):
            print("This donor face is not present in the database, so there's nothing to delete")
        else:
            delete_entry(name)
            print("Face deleted!")
    elif operation == "5":
        face = input("Enter in the name of the face you would like to see: ")
        while face == "" or not get_face(face):
            face = input("This face is not present in the database, please try again: ")
        print("Here is the face you asked for.")  # Rating information will be included here when it's done
        cv2.imshow("Viewing face: " + face, convert_image(get_face(face)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # elif operation == "6":
    #     oldName = input("Enter in the name of the face you would like to rename: ")
    #     while not get_face_sql(oldName):
    #         oldName = input("This face is not present in the database, please try again: ")
    #     newName = input("What would you like to rename " + oldName + " to? ")
    #     dbFace[newName] = dbFace.pop(oldName)
    #     dbPoints[newName] = dbPoints.pop(oldName)
    #     dbRatings[newName] = dbRatings.pop(oldName)
    #     print(oldName + " renamed to " + newName)
    # elif operation == "7":
    #     filepath = input("Enter in the file path of the face that will replace a face: ")
    #     while not os.path.isfile(filepath):
    #         filepath = input("Invalid file, try again: ")
    #     name = input("Enter in the name for the face which should be replaced : ")
    #     while not get_face_sql(name):
    #         name = input("This face is not present in the database, please try again: ")
    #     img = cv2.imread(filepath)
    #     points = precalculate_face(img)
    #     if not points:
    #         print("No face detected in this image")
    #     else:
    #         # dbFace[name] = img  # Dlib and OpenCV faces are not compatible color-wise, one must use BGR
    #         dbPoints[name] = points
    #         dbRatings[name] = 0
    #         print("Face replaced!")
    # elif operation == "8":
    #     face = input("Enter in the name of the face you would like to flip: ")
    #     while not get_face_sql(face):
    #         face = input("This face is not present in the database, please try again: ")
    #     # img = cv2.flip(dbFace[face], 1)
    #     # dbFace[face] = img
    #     # dbPoints[face] = precalculate_face(img)
    #     dbRatings[face] = 0
    else:
        print("Invalid operation...goodbye")
        # connection_neo4j.close()
        exit()


# From nats.py/examples/service.py
async def run(loop):
    nc = NATS()

    # Note: these callbacks must be inside an async or they will not reply properly
    async def create_entry_nats(msg):
        name = msg.data.decode()
        print("Creating face", name)
        img = get_face(name)  # lena.jpg takes 768.16 KB as a cv2 image vs 28.67 KB as a file to store
        points = precalculate_face(convert_image(img))
        if not points:
            print("No face detected in this image")
            client_minio.remove_object(buckets["faces"], name)
            await nc.publish(msg.reply, b'No face detected')
        else:
            create_entry(name, img, points)
            print(name + " added!")
            await nc.publish(msg.reply, b'Added')

    async def list_entries_nats(msg):
        list = get_entry_list()
        await nc.publish(msg.reply, b'' + str(list).encode("utf-8"))
        print("Listing nodes")

    async def swap_nats(msg):
        name_combined = msg.data.decode()
        donor, base = name_combined.split("|")
        if is_entry(donor) is None or is_entry(base) is None:
            await nc.publish(msg.reply, str("").encode("utf-8"))
        else:
            output = swap_face_calc(convert_image(get_face(donor)), get_points(donor),
                                    convert_image(get_face(base)), get_points(base))
            update_face(name_combined, output)
            await nc.publish(msg.reply, str(name_combined).encode("utf-8"))
            print("Swapping", name_combined)

    async def delete_entries_nats(msg):
        name = msg.data.decode()
        if not is_entry(name):
            print("No need to remove " + name + " because it's gone")
            await nc.publish(msg.reply, b'Not present')
        else:
            delete_entry(name)
            await nc.publish(msg.reply, b'Deleted')
            print("Deleted", name)

    async def view_nats(msg):
        name = msg.data.decode()
        if is_entry(name) is None:
            await nc.publish(msg.reply, str("").encode("utf-8"))
        else:
            base_rating = get_rating_base(name)
            donor_rating = get_rating_donor(name)
            await nc.publish(msg.reply, str("base:" + str(base_rating) + " donor:" + str(donor_rating)).encode("utf-8"))
            print("Viewing", name)

    async def rank_entries_nats(msg):
        name_combined = msg.data.decode()
        donor, base, liked = name_combined.split("|")
        if is_entry(donor) is None or is_entry(base) is None:
            await nc.publish(msg.reply, str("error").encode("utf-8"))
        else:
            await nc.publish(msg.reply, str("rating is now" + str(update_rating(donor, base, liked == "1"))).encode("utf-8"))

    async def closed_cb():
        print("Connection to NATS is closed.")
        await asyncio.sleep(0.1, loop=loop)
        loop.stop()

    # It is very likely that the demo server will see traffic from clients other than yours.
    # To avoid this, start your own locally and modify the example to use it.
    options = {
        "servers": [NATS_ADDRESS],
        # "servers": ["nats://demo.nats.io:4222"],
        "loop": loop,
        "closed_cb": closed_cb
    }

    await nc.connect(**options)
    print(f"Connected to NATS at {nc.connected_url.netloc}...")

    # Basic subscription to receive all published messages
    # which are being sent to a single topic 'discover'
    await nc.subscribe("create", cb=create_entry_nats)
    await nc.subscribe("list", cb=list_entries_nats)
    await nc.subscribe("swap", cb=swap_nats)
    await nc.subscribe("delete", cb=delete_entries_nats)
    await nc.subscribe("view", cb=view_nats)
    await nc.subscribe("rank", cb=rank_entries_nats)

    def signal_handler():
        if nc.is_closed:
            return
        print("Disconnecting...")
        loop.create_task(nc.close())

    for sig in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(getattr(signal, sig), signal_handler)

    print("Listening for requests...")
    for i in range(1, 1000000):
        await asyncio.sleep(1)


class Neo4jTx:
    @staticmethod
    def transaction(cql):
        with connection_neo4j.session() as client_neo4j:
            result = client_neo4j.write_transaction(cql)
        return result

    @staticmethod
    def transaction1(cql, name):
        with connection_neo4j.session() as client_neo4j:
            result = client_neo4j.write_transaction(cql, name)
        return result

    @staticmethod
    def transaction2(cql, old_name, new_name):
        with connection_neo4j.session() as client_neo4j:
            result = client_neo4j.write_transaction(cql, old_name, new_name)
        return result

    @staticmethod
    def transaction3(cql, old_name, new_name, value):
        with connection_neo4j.session() as client_neo4j:
            result = client_neo4j.write_transaction(cql, old_name, new_name, value)
        return result

    @staticmethod
    def create_face(tx, name):
        tx.run("CREATE (n:Face {name:'" + name + "'})")

    @staticmethod
    def delete_face(tx, name):
        tx.run("match (n:Face) where n.name='" + name + "' detach delete n")

    @staticmethod
    def rename_face(tx, old_name, new_name):
        tx.run("match (n:Face) where n.name='" + old_name + "' set n.name='" + new_name + "'")

    @staticmethod
    def list_face(tx):
        return tx.run("match (n:Face) return n as Node").data()

    @staticmethod
    def rate_face(tx, donor_name, base_name, liked):
        if liked:
            rating = "+"
        else:
            rating = "-"
        result = tx.run("match(n: Face) where n.name = '" + donor_name + "'" +
                        " optional match(o: Face) where o.name = '" + base_name + "'" +
                        " merge(n) - [r: Swap]->(o)" +
                        " set r.rating = coalesce(r.rating, 0) " + rating + " 1" +
                        " return r.rating as rating")
        values = []
        for record in result:
            values.append(record.values())
        return values[0]  # there should only be one rating returned

    @staticmethod
    def get_rating_donor(tx, name):
        return tx.run("match (n)-[r:Swap]->() where n.name='" + name + "' return sum(r.rating)").values()

    @staticmethod
    def get_rating_base(tx, name):
        return tx.run("match ()-[r:Swap]->(n) where n.name='" + name + "' return sum(r.rating)").values()


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


def set_object(name, bucket_name, object_source):
    object_pickled = pickle.dumps(object_source)
    client_minio.put_object(buckets[bucket_name], name, io.BytesIO(object_pickled), len(object_pickled))


def create_entry(name, img, points):
    # update_face(name, img)
    update_points(name, points)
    Neo4jTx.transaction1(Neo4jTx.create_face, name)


def create_entry_path(name, path):
    client_minio.fput_object(buckets["faces"], name, path)
    Neo4jTx.transaction1(Neo4jTx.create_face, name)


def delete_entry(name):
    for bucket in buckets.keys():
        client_minio.remove_object(buckets[bucket], name)
    Neo4jTx.transaction1(Neo4jTx.delete_face, name)


def get_entry_list(): return Neo4jTx.transaction(Neo4jTx.list_face)


def get_face(name):
    data = get_object(name, "faces")
    if type(data) == bool and not data:  # TODO: use the graph store instead of the object store to test if present
        return False
    else:
        return pickle.loads(data)


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


def update_face(name, img): set_object(name, "faces", img)


def get_rating_base(name): return Neo4jTx.transaction1(Neo4jTx.get_rating_base, name)


def get_rating_donor(name): return Neo4jTx.transaction1(Neo4jTx.get_rating_donor, name)


# TODO: make the "rating" of a face the total sum of the weights of its outward edges
# def get_rating(name):
#     result = Neo4jTx.transaction1(Neo4jTx.get_rating, name)
#     value = [record["rating"] for record in result]
#     return value[0]["rating"]


def update_rating(donor, base, liked): return Neo4jTx.transaction3(Neo4jTx.rate_face, donor, base, liked)


def update_name(old_name, new_name): Neo4jTx.transaction2(Neo4jTx.rename_face, old_name, new_name)


def get_points(name): return pickle.loads(get_object(name, "points"))


def update_points(name, points): set_object(name, "points", points)


def read_image(path):
    with open(path, "rb") as fd:
        return fd.read()


def convert_image(img) -> np.ndarray:
    """
    Convert a blob read through fd.read() and turn it into a numpy image.
    :param img: An image retrieved from get_face_sql() or from the disk via .read()ing a file descriptor.
    :return: A numpy version of img.
    """
    return cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)


if __name__ == '__main__':
    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    if int(major_ver) < 3:
        print(sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    # TODO: should we keep the connection open for the duration of the program/microservice or open/shut it when needed?
    # Set up Neo4j for storing face names, rankings, and other metadata
    connection_neo4j = GraphDatabase.driver(NEO4J_ADDRESS, auth=("neo4j", "password"))  # can't use default password or neo4j will get upset, so make sure to reconfigure neo4j before starting
    # client_neo4j = connection_neo4j.session()
    # We can't create more than one user database with the community version of neo4j so we'll just use the default one

    # Set up MinIO for storing face images and point-position tuples
    client_minio = Minio(
        MINIO_ADDRESS,
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

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    try:
        loop.run_forever()
    finally:
        loop.close()
    # while True:
    #     demo_prompt()
