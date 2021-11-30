
import numpy as np
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description='Average frames of a video into a single image')
parser.add_argument('-i', help='Input filename')
parser.add_argument('-o', help='Output filename')
timestr = time.strftime("%Y%m%d-%H%M%S")
(rAvg, gAvg, bAvg) = (0, 0, 0)


def run(input_file):
    global rAvg
    global gAvg
    global bAvg    
    cap = cv2.VideoCapture(input_file)
    total = 0
    

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        print("frame: " + f'{total + 1}')

        if ret == True:
            #split bgr
            (B, G, R) = cv2.split(frame.astype("float"))

            if rAvg is None:
                rAvg = R
                bAvg = B
                gAvg = G

            else:  # below here the operations for the full r g b channels per image
                rAvg = ((total * rAvg) + (1 * R)) / (total + 1.1)
                gAvg = ((total * gAvg) + (1 * G)) / (total + 1.1)
                bAvg = ((total * bAvg) + (1 * B)) / (total + 1.1)
            total += 1

            # merge the RGB averages together and write the output image to disk
        else:
            break
    
    cap.release()




if __name__ == '__main__':
    args = parser.parse_args()
    run(args.i)
    avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")
    cv2.imwrite(timestr + args.o, avg)
