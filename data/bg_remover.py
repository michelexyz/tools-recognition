import sys
from pathlib import Path
from rembg import remove, new_session
import os, shutil
import cv2 as cv


session = new_session()

rawFolderStr = '/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/raw'
rawFolder = Path(rawFolderStr)
#processedFolderStr = str(rawFolder.parent / 'processed')
processedFolderStr = '/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed'


# for filename in os.listdir(processedFolderStr):
#     file_path = os.path.join(processedFolderStr, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))
# print('file eliminati')
print('ciao')
print('1')
print('2')
print('3')
for category in os.listdir(rawFolderStr):
    categoryPathStr = rawFolderStr + '/' + category
    categoryPath = Path(categoryPathStr)
    processedCategoryPathStr = processedFolderStr + '/' + category
    processedCategoryPath = Path(processedCategoryPathStr)
    processedCategoryPath.mkdir(parents=True,exist_ok=True)
    for file in categoryPath.glob('*'):
        input_path = str(file)

        imgName = (file.stem + ".out.png")
        output_path = str(processedFolderStr + '/' + category + '/' + imgName)

        input = cv.imread(input_path)
        noBgImg = remove(input, session=session)

        #Contours,imgContours = cv.findContours(noBgImg,None , None)
        [X, Y, W, H] = cv.boundingRect(cv.cvtColor(noBgImg, cv.COLOR_BGR2GRAY))
        cropped_image = noBgImg[Y:Y + H, X:X + W]

        cv.imwrite(output_path, cropped_image)
        print('generata ' + imgName)



