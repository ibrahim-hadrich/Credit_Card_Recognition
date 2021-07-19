import tkinter as tk
from cv2 import cv2
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image
import imutils
from imutils import contours
import numpy as np
class Page(tk.Tk):

    def __init__(self):

        tk.Tk.__init__(self)
        self.title("credit card detection")
        
        container=tk.Frame(self)
        container.grid()

        # select image                          
        self.browse_lbl = tk.Label(self, text="Select Image :", font=("Arial", 10), fg="#202020")
        self.browse_lbl.grid(row=4, column=0, columnspan=3, padx=24, pady=10, sticky="w")

        self.browse_entry=tk.Entry(self, text="", width=30)
        self.browse_entry.grid(row=4, column=0, columnspan=3, padx=120, pady=10, sticky="w")

        self.browse_btn = tk.Button(self, text="Browse", bg="#ffffff", relief="flat", width=10,
                                    command=lambda:self.show_image())
        self.browse_btn.grid(row=4, column=0, padx=310, pady=10, columnspan=3, sticky="w")
        
        # show image
        self.lbl_image = tk.Label(self, image="")
        self.lbl_image.grid(row=8, column=0, pady=25, padx=10, columnspan=3, sticky="nw")

        # status text
        self.label_text_progress = tk.StringVar()
        self.scan_progress = tk.Label(self, textvariable=self.label_text_progress, font=("Arial", 10),fg="#0000ff")
        
        # scan button
        self.scan_btn = tk.Button(self, text="Process", bg="#ffffff", relief="flat",
                                 width=10, command=lambda:self.ocr())
       
    def show_image(self):
        global path
        global path1
        
        # open file dialog
        self.path = filedialog.askopenfilename(defaultextension="*.png", filetypes = (("PNG","*.png"),("JPG", "*.jpg")))
        self.browse_entry.delete(0, tk.END)
        self.browse_entry.insert(0, self.path)
        
        self.label_text_progress.set("Image loaded - ready to be processed.")
        self.scan_progress.grid(row=18, column=0, padx=10, pady=0,
                           columnspan=3, sticky="w")
    
        # resize image
        cv_img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        height, width, no_channels = cv_img.shape

        HEIGHT = 400 
        imgScale = HEIGHT/height
        newX, newY = cv_img.shape[1]*imgScale, cv_img.shape[0]*imgScale
        newimg = cv2.resize(cv_img, (int(newX), int(newY)))
        photo = ImageTk.PhotoImage(image = Image.fromarray(newimg))
        
        # show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo

        # add scan button
        scan_btn_mid = int(newX/2) - 40;       
        self.scan_btn.grid(row=17, column=0, padx=scan_btn_mid, pady=10,
                           columnspan=3, sticky="w")

    def ocr(self):
        FIRST_NUMBER = {
	            "3": "American Express",
	            "4": "Visa",
	            "5": "MasterCard",
	            "6": "Discover Card"
            }
        # load ref and convert to gray
        ocr_ref = cv2.imread("C:/Users/Ibrahim/Desktop/PIITMW/sem2/projectPythonOCR/ocr_a_reference.png")
        ocr_ref = cv2.cvtColor(ocr_ref, cv2.COLOR_BGR2GRAY)
        ocr_ref= cv2.threshold(ocr_ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
        # find contours in the OCR-A image (i.e,. the outlines of the digits)
        # sort them from left to right, and initialize a dictionary to map
        # digit name to the ROI
        refCnts = cv2.findContours(ocr_ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
        digits = {} 
        # loop over the OCR-A reference contours
        for (i, c) in enumerate(refCnts):
            # compute the bounding box for the digit, extract it, and resize
            # it to a fixed size
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ocr_ref[y:(y + h), x:(x + w)]
            # update the digits dictionary, mapping the digit name to the ROI
            digits[i] = roi
            # initialize a rectangular (wider than it is tall) and square
        # structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # load the input image, resize it, and convert it to grayscale
        ocr_image = cv2.imread(self.path)
        ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_RGB2BGR)
        ocr_image = imutils.resize(ocr_image, width=300)
        gray = cv2.cvtColor(ocr_image, cv2.COLOR_RGB2GRAY)
        #cv2.imshow('image',ocr_image)
        #make the image black
        black = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
        #cv2.imshow("tophat",black)
        # compute the Scharr gradient of the black image, then scale
        # the rest back into the range [0, 255]
        gradX = cv2.Sobel(black, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")
        #cv2.imshow("gradx",gradX)
        # apply a closing operation using the rectangular kernel to help
        # cloes gaps in between credit card number digits, then apply
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # apply a second closing operation to the binary image, again
        # to help close gaps between credit card number regions
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        #cv2.imshow("thresh",thresh)
        # find contours in the thresholded image, then initialize the
        # list of digit locations
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        locs = []
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # compute the bounding box of the contour, then use the
            # bounding box coordinates to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # since credit cards used a fixed size fonts with 4 groups
            # of 4 digits, we can prune potential contours based on the
            # aspect ratio
            if ar > 2.5 and ar < 4.5:
                # contours can further be pruned on minimum/maximum width
                # and height
                if (w > 40 and w < 70) and (h > 10 and h < 20):
                    # append the bounding box region of the digits group
                    # to our locations list
                    locs.append((x, y, w, h))
        # sort the digit locations from left-to-right, then initialize the
        # list of classified digits
        locs = sorted(locs, key=lambda x:x[0])
        #print(locs)
        output = []
        # loop over the 4 groupings of 4 digits
        for (i, (gX, gY, gW, gH)) in enumerate(locs):
            # initialize the list of group digits
            groupOutput = []
        
            # extract the group ROI of 4 digits from the grayscale image,
            # then apply thresholding to segment the digits from the
            # background of the credit card
            group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
            group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
            # detect the contours of each individual digit in the group,
            # then sort the digit contours from left to right
            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            
            digitCnts = digitCnts[1] if imutils.is_cv3() else digitCnts[0]
            digitCnts = contours.sort_contours(digitCnts,
            method="left-to-right")[0]

            # loop over the digit contours
            for c in digitCnts:
                # compute the bounding box of the individual digit, extract
                # the digit, and resize it to have the same fixed size as
                # the reference OCR-A images
                (x, y, w, h) = cv2.boundingRect(c)
                roi = group[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))
        
                # initialize a list of template matching scores	
                scores = []
        
                # loop over the reference digit name and digit ROI
                for (digit, digitROI) in digits.items():
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    result = cv2.matchTemplate(roi, digitROI,
                                            cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)
        
                # the classification for the digit ROI will be the reference
                # digit name with the *largest* template matching score
                groupOutput.append(str(np.argmax(scores)))
                # draw the digit classifications around the group
                cv2.rectangle(ocr_image, (gX - 5, gY - 5),
                            (gX + gW + 5, gY + gH + 5), (0, 255, 0), 2)
                cv2.putText(ocr_image, "".join(groupOutput), (gX, gY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
            # update the output digits list
            output.extend(groupOutput)
        # display the output credit card information to the screen
        imagerect=cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("resultat",imagerect)
        tk.messagebox.showinfo('credit card information',"Credit Card Type: {}".format(FIRST_NUMBER[output[0]])+
            "\n "+"Credit Card #: {}".format("".join(output)))

if __name__ == "__main__":
    app = Page()
    app.geometry("700x725+100+100")
    app.mainloop()