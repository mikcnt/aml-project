import PySimpleGUI as sg
import os.path
from PIL import Image, ImageTk
from model import ColorizationNet
from utils import *

img_size = (350, 350)
img_box_size = (800, 350)
image_orig_str = "-IMAGE_ORIG-"
image_pred_str = "-IMAGE-PRED-"


def get_img(filename):
    """ Generate png image from jpg """
    img = Image.open(filename).resize(img_size)
    return ImageTk.PhotoImage(img)


def get_img_prediction(model, pathname):
    img = np.array(Image.open(pathname).convert('RGB'))
    img_original_size = img.shape[:2]
    img_lab = rgb2lab(img)
    img_gray = img_lab[:, :, 0] / 100
    img_gray_small = resize(img_gray, (224, 224))
    img_gray_tensor = torch.from_numpy(img_gray_small).unsqueeze(0).float()
    img_gray_batch = img_gray_tensor.unsqueeze(0)
    img_smooth = model(img_gray_batch)[0]

    img_lab = gray_smooth_tensor2lab(img_gray_small, img_smooth)
    img_lab_resized = resize(img_lab, img_original_size)
    img_gray = img_gray * 100
    img_lab_resized[:, :, 0] = img_gray
    img_rgb = resize(lab2rgb(img_lab_resized), img_original_size)
    # img_prediction = gray_smooth_tensor2rgb(img_gray_tensor, img_smooth.detach())
    img_from_array = Image.fromarray((img_rgb * 255).astype(np.uint8)).resize(img_size)

    return ImageTk.PhotoImage(image=img_from_array)


layout = [[sg.Text("Automatic Image Colorization")]]

file_list_column = [
    [
        sg.Text("Select model"),
        sg.In(size=(25, 1), enable_events=True, key="-MODEL-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ]
]

image_viewer_column_original = [
    [sg.Text("True Image")],
    [sg.Image(size=img_size, key=image_orig_str)]
]

image_viewer_column_pred = [
    [sg.Text("Predicted Colorization")],
    [sg.Image(size=img_size, key=image_pred_str)]
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column_original),
        sg.Column(image_viewer_column_pred)
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
               and f.lower().endswith((".jpg", ".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-MODEL-":
        # a model has been selected
        # load model
        checkpoint = values["-MODEL-"]
        checkpoint = torch.load(checkpoint)
        model = ColorizationNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model correctly loaded")
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        # try:
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
        window[image_orig_str].update(data=get_img(filename))
        window[image_pred_str].update(data=get_img_prediction(model, filename))

        # except Exception as e:
        #    print(e)

window.close()
