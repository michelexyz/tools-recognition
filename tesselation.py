import cv2 as cv

from tr_utils.useful import *


# divides the input image in sub-regions and describes them with the passed function
def tassella_e_descrivi(img, step, dim, num_features, descriptor_funct=None):

    # tassello number
    num = 0

    # img dimensions
    width = img.shape[1]
    height = img.shape[0]

    # computing taxels 'img' dimensions
    rows_n = ((height-dim) // step) + 1
    columns_n = ((width-dim) // step) + 1

    descripted_img_size = rows_n * columns_n
    print('michele gay')
    # print('numero colonne: ' + str(columns_n) + 'numero righe: ' + str(rows_n))
    print('array creato')
    descripted_img = np.empty((descripted_img_size,num_features), dtype=np.float32)


    # iterating over image
    for i in range(0, height-(dim+1), step):

        for j in range(0, width-(dim+1), step):

            point = (j, i)
            roi = extract_roi(img, point, dim, dim)
            print('tassello numero ' + str(num) + ' estratto')

            # applico il descrittore al tassello (ROI)
            # but only if the function is not None
            if descriptor_funct is not None:
                descripted_roi = descriptor_funct(roi)
                print('tassello numero '+str(num)+' descripted')
            else:
                # ATTENTION! in this case, the roi is not beeing descripted despite its name
                descripted_roi = roi
                print('ooh va che non stai descrivendo niente fratello')

            # put the computed (descripted) roi in descripted_img matrix
            descripted_img[num] = descripted_roi

            # resize_and_show('roi numero ' + str(num), roi, 190)

            num += 1

    return descripted_img


def tassella_e_descrivi_png(img_path, step, dim, num_features, descriptor_funct=None):

    # tassello number
    num = 0

    # Carica immagine .png nei 4 canali
    img_png = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    img_rgb = cv.imread(img_path)

    # split png img in channels
    B, G, R, A = cv.split(img_png)

    # BINARIZE IMAGE
    binarized_bool = (A > 0).astype("uint8")

    # TRANSFORM OPERATIONS
    # image is an rgb image with only the biggest object, the rest has value 0
    binarized_bool, image = remove_imperfections_adv(binarized_bool, img_rgb)

    # use 'image' for description
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # img dimensions
    width = gray.shape[1]
    height = gray.shape[0]

    # print('numero colonne: ' + str(columns_n) + 'numero righe: ' + str(rows_n))
    descripted_img = []
    print('lista creata')

    # iterating over image
    for i in range(0, height-(dim+1), step):

        for j in range(0, width-(dim+1), step):

            point = (j, i)
            roi = extract_roi(gray, point, dim, dim)
            print('tassello numero ' + str(num) + ' estratto')

            # descrivo il tassello SOLO se è accettabile
            accettabile = good_tassello(roi)

            if accettabile:
                # applico il descrittore al tassello (ROI)
                # but only if the function is not None
                if descriptor_funct is not None:
                    descripted_roi = descriptor_funct(roi)
                    print(f'tassello numero {num} descripted')
                else:
                    # ATTENTION! in this case, the roi is not beeing descripted despite its name
                    descripted_roi = roi
                    print('ooh va che non stai descrivendo niente fratello')

                # put the computed (descripted) roi in descripted_img list
                descripted_img.append(descripted_roi)

                # resize_and_show('roi numero ' + str(num), roi, 190)

            else:
                # no good tassello: scartato
                print(f'tassello numero {num} SCARTATO')

            num += 1

    return descripted_img


# if there is too much black area, return false: not good tassello
def good_tassello(tassello):
    limit_black = 20
    black_area = zero_pixel_number(tassello)

    if black_area > limit_black:
        # tassello ain't good enough
        return False
    else:
        # tassello is good
        return True

def zero_pixel_number(img):

    black_counter = 0
    tot_counter = 0

    height = img.shape[0]
    width = img.shape[1]

    # iterating on image
    for i in range(0, height):
        for j in range(0, width):

            # check if is black
            if img[j,i] == 0:
                black_counter += 1
            tot_counter += 1

    # compute percentage
    black_perc = int((black_counter / tot_counter) * 100)

    return black_perc



# example function, writes a number on an image
def numera(img, numero):
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (img.shape[0]//2,img.shape[1]//2)
    new_img = cv.putText(img, str(numero), org, font, 1, (255,0,0), 1, cv.LINE_AA)
    return new_img

# immagine = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/tools.jpg')
# dim = (400, 400)
# resized = cv.resize(immagine, dim, interpolation=cv.INTER_AREA)
# resize_and_show('roi', resized)
#
# step = 40
# size = 50
#
# # function to be passed in 'tassellamela'
# # this funct, will be applied to all rois
# funct = numera
#
# tassellamela(resized, step, size, funct)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
