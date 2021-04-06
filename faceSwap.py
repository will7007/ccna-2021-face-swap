#! /usr/bin/env python
# This code is my combination of https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
# (some old face swap code which used hard-coded points)
# and http://dlib.net/face_alignment.py.html (dlib face allignment example, so we can get facial points for any image)

import sys
import numpy as np
import cv2
import dlib
import os
import mysql.connector


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

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


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) # MUST be a 2-element tuple to not throw here

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
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect


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


def demoprompt():
    print("\n\nCCNA Project Group 1: Face Swapping")
    print("List of possible operations:")
    print("1: Create face in database")
    print("2: List faces in database")
    print("3: Take a face in the database and combine it with a head in the database")
    print("4: Delete a face from the database")
    print("5: View a face from the database")
    print("6: Rename a face from the database")
    print("7: Change the face assigned to a name in the database")
    print("8: Y-axis flip a face for improved results")
    operation = input("Enter in an operation number [1-5]")
    if operation == "1":
        filepath = input("Enter in the file path of the face you would like to store: ")
        while not os.path.isfile(filepath):
            filepath = input("Invalid file, try again: ")
        name = input("Enter in a name for the face you just selected: ")
        if get_face_sql(name):
            print("Sorry, " + name + " is already being used in the database")
        else:
            # img = cv2.imread(filepath)
            img = read_image(filepath)
            points = precalculate_face(convert_image(img))  # todo: find a more efficient way of doing reading + storing
            if not points:
                print("No face detected in this image")
            else:
                # dbFace[name] = img  # Dlib and OpenCV faces are not compatible color-wise, one must use BGR
                set_face_sql(name, img)
                dbPoints[name] = points
                # dbRatings[name] = 0
                print("Face added!")
    elif operation == "2":
        print("Here is a list of faces currently in the database:")
        print(get_face_list())
    elif operation == "3":
        donor = input("Enter in the donor face which will be transplanted to the base face: ")
        while not get_face_sql(donor):
            donor = input("This donor face is not present in the database, please try again: ")
        base = input("Enter in the base face which will have its face overwritten by the donor face: ")
        while not get_face_sql(base):
            base = input("This base face is not present in the database, please try again: ")
        cv2.imshow("Face Swapped", swap_face_calc(convert_image(get_face_sql(donor)), dbPoints[donor],
                                                  convert_image(get_face_sql(base)), dbPoints[base]))
        print("Resulting person has been displayed in a window!")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rating = input("Was that face swap good? Enter in Y or y if it was or anything else if it wasn't.")
        if rating == "Y" or rating == "y":
            print("Great, glad you liked it!")
            set_ranking_sql(donor, True)
            set_ranking_sql(base, True)
        else:
            print("Oh, sorry about that!")
            set_ranking_sql(donor, False)
            set_ranking_sql(base, False)
    elif operation == "4":
        name = input("Enter in the face which will be removed from the database: ")
        if not get_face_sql(name):
            print("This donor face is not present in the database, so there's nothing to delete")
        else:
            delete_face_sql(name)
            dbPoints.pop(name)
            print("Face deleted!")
    elif operation == "5":
        face = input("Enter in the name of the face you would like to see: ")
        while not get_face_sql(face):
            face = input("This face is not present in the database, please try again: ")
        print("Here is the face you asked for. It has a rating of", get_ranking_sql(face))
        cv2.imshow("Viewing face: " + face, convert_image(get_face_sql(face)))
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
        exit()


def set_face_sql(name, img):
    if len(name) < 128:
        query = "insert into faces (Name, Ranking, Photo) values (%s,%s,%s)"  # Seems to be no equivalent to OUTPUT in MySQL for PKs
        values = (name, 0, img)
        cursor.execute(query, values)
        conn.commit()
        cursor.fetchall()  # todo: may not need this


def get_face_sql(name):
    cursor.execute("select Photo from faces where Name=%s", (name,))
    result = cursor.fetchone()
    if result is None:
        return False
    else:
        return result[0]


def get_face_list():
    cursor.execute("select Name, Ranking from faces")
    return cursor.fetchall()


def get_ranking_sql(name):
    cursor.execute("select Ranking from faces where Name=%s",(name,))
    return cursor.fetchone()


def set_ranking_sql(name, liked):
    if liked:
        cursor.execute("update faces set Ranking=Ranking+1 where Name=%s", (name,))
    else:
        cursor.execute("update faces set Ranking=Ranking-1 where Name=%s", (name,))


def delete_face_sql(name):
    cursor.execute("delete from faces where Name=%s", (name,))


def update_name_sql(old_name, new_name):
    query = "update faces set Name=%s where Name=%s"
    values = (new_name, old_name)
    cursor.execute(query, values)
    cursor.fetchall()  # todo: may not need this


def update_face_sql(name, img):
    query = "update faces set Photo=? where Name=?"
    values = (img, name)
    cursor.execute(query, values)
    conn.commit()
    cursor.fetchall()  # todo: may not need this


def read_image(path):
    with open(path, "rb") as fd:
        return fd.read()


def convert_image(img) -> np.ndarray:
    """
    Convert an blob that has been stored in SQL through fd.read() and turn it into a numpy image.
    :param img: An image retrieved from get_face_sql() or from the disk via .read()ing a file descriptor.
    :return: A numpy version of img.
    """
    return cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)


if __name__ == '__main__':
    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    if int(major_ver) < 3 :
        print(sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    conn = mysql.connector.connect(host='localhost', database='mysql', user='root', password='password')
    cursor = conn.cursor()
    cursor.execute("create database if not exists face_swap;")
    cursor.execute("use face_swap;")
    cursor.execute("create table if not exists faces(Name varchar(255) not null, ID int auto_increment,"
                   "Ranking int not null, Photo longblob not null, primary key (ID))")
    cursor.execute("create table if not exists facial_points(points blob, Face_ID int not null,"
                   "foreign key (Face_ID) references face_swap.faces(ID) on update cascade on delete cascade);")
    # Blob of tuple points aren't ideal, but it works: eval(str(futureTuple)[3:-3])
    # cursor.execute("insert into faces (Name, Photo) values(%s,%s)", ('Lena', img,))
    # cursor.execute("insert into facial_points (points, Face_ID) values('((12,34),(56,78))',1)")


    # Start MySQL's container for this script
    # docker run --name mysql --network host -e MYSQL_ROOT_PASSWORD=password mysql

    # https://stackoverflow.com/questions/66663132/valueerror-buffer-size-must-be-a-multiple-of-element-size-when-converting-from

    # These will be replaced by a SQL database/similar
    # dbFace = {}  # Stores the numpy array of the face image
    dbPoints = {}  # Stores the precalculated facial landmarks  # TODO: fully switch to using DB for storing facial points
    # FIXME: we cannot store weighted relations easily in SQL, should we use Neo4j instead?
    # dbRatings = {}  # Stores what people think of each face

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    while True:
        demoprompt()
