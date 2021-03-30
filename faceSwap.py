#! /usr/bin/env python
# This code is my combination of https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
# (some old face swap code which used hard-coded points)
# and http://dlib.net/face_alignment.py.html (dlib face allignment example, so we can get facial points for any image)

import sys
import numpy as np
import cv2
import dlib
import os


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) # MUST be a 2-element tuple to not throw here
    
    triangleList = subdiv.getTriangleList();
    
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
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

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
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

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
        if name in dbFace.keys():
            print("Sorry, " + name + " is already being used in the database")
        else:
            img = cv2.imread(filepath)
            points = precalculate_face(img)
            if not points:
                print("No face detected in this image")
            else:
                dbFace[name] = img  # Dlib and OpenCV faces are not compatible color-wise, one must use BGR
                dbPoints[name] = points
                dbRatings[name] = 0
                print("Face added!")
    elif operation == "2":
        print("Here is a list of faces currently in the database:")
        for face in dbFace.keys():
            print("Name:", face, "| Rating:", dbRatings[face])
            # In the future, we could include more info here about how this face has been rated (if that is added)
            # And how many times it has been used
        if len(dbFace.keys()) == 0:
            print("No faces currently inside the database...")
    elif operation == "3":
        donor = input("Enter in the donor face which will be transplanted to the base face: ")
        while donor not in dbFace.keys():
            donor = input("This donor face is not present in the database, please try again: ")
        base = input("Enter in the base face which will have its face overwritten by the donor face: ")
        while base not in dbFace.keys():
            base = input("This base face is not present in the database, please try again: ")
        cv2.imshow("Face Swapped", swap_face_calc(dbFace[donor], dbPoints[donor], dbFace[base], dbPoints[base]))
        print("Resulting person has been displayed in a window!")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rating = input("Was that face swap good? Enter in Y or y if it was or anything else if it wasn't.")
        if rating == "Y" or rating == "y":
            print("Great, glad you liked it!")
            dbRatings[donor] += 1
            dbRatings[base] += 1
        else:
            print("Oh, sorry about that!")
            dbRatings[donor] -= 1
            dbRatings[base] -= 1
    elif operation == "4":
        name = input("Enter in the face which will be removed from the database: ")
        if name not in dbFace.keys():
            print("This donor face is not present in the database, so there's nothing to delete")
        else:
            dbFace.pop(name)
            dbPoints.pop(name)
            dbRatings.pop(name)
            print("Face deleted!")
    elif operation == "5":
        face = input("Enter in the name of the face you would like to see: ")
        while face not in dbFace.keys():
            face = input("This face is not present in the database, please try again: ")
        print("Here is the face you asked for. It has a rating of", dbRatings[face])
        cv2.imshow("Viewing face: " + face, dbFace[face])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif operation == "6":
        oldName = input("Enter in the name of the face you would like to rename: ")
        while oldName not in dbFace.keys():
            oldName = input("This face is not present in the database, please try again: ")
        newName = input("What would you like to rename " + oldName + " to? ")
        dbFace[newName] = dbFace.pop(oldName)
        dbPoints[newName] = dbPoints.pop(oldName)
        dbRatings[newName] = dbRatings.pop(oldName)
        print(oldName + " renamed to " + newName)
    elif operation == "7":
        filepath = input("Enter in the file path of the face that will replace a face: ")
        while not os.path.isfile(filepath):
            filepath = input("Invalid file, try again: ")
        name = input("Enter in the name for the face which should be replaced : ")
        while name not in dbFace.keys():
            name = input("This face is not present in the database, please try again: ")
        img = cv2.imread(filepath)
        points = precalculate_face(img)
        if not points:
            print("No face detected in this image")
        else:
            dbFace[name] = img  # Dlib and OpenCV faces are not compatible color-wise, one must use BGR
            dbPoints[name] = points
            dbRatings[name] = 0
            print("Face replaced!")
    elif operation == "8":
        face = input("Enter in the name of the face you would like to flip: ")
        while face not in dbFace.keys():
            face = input("This face is not present in the database, please try again: ")
        img = cv2.flip(dbFace[face], 1)
        dbFace[face] = img
        dbPoints[face] = precalculate_face(img)
        dbRatings[face] = 0
    else:
        print("Invalid operation...goodbye")
        exit()


if __name__ == '__main__' :
    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3 :
        print(sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    # These will be replaced by a SQL database/similar
    dbFace = {}  # Stores the numpy array of the face image
    dbPoints = {} # Stores the precalculated facial landmarks
    dbRatings = {} # Stores what people think of each face

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    while True:
        demoprompt()
