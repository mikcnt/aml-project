import PySimpleGUI as sg
import os.path
from model import ColorizationNet
from utils import *
import torch

img_size = (350, 350)
img_box_size = (800, 350)
image_orig_str = "-IMAGE_ORIG-"
image_pred_str = "-IMAGE-PRED-"


layout = [[sg.Text("Automatic Image Colorization")]]

file_list_column = [
    [
        sg.Text("Select loss type"),
        sg.DropDown(['classification', 'regression'], key="-LOSS-", enable_events=True),
    ],
    [
        sg.Text("Select model"),
        sg.In(size=(25, 1), enable_events=True, key="-MODEL-", disabled=True),
        sg.FileBrowse(disabled=True, key='-MODEL_BROWSE-'),
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
        model = ColorizationNet(values["-LOSS-"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model correctly loaded")
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        # try:
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
        window[image_orig_str].update(data=get_img(filename))

        img_pred_tk = get_img_prediction_as_tk(model, filename, img_size)
        window[image_pred_str].update(data=img_pred_tk)

        # except Exception as e:
        #    print(e)
    elif event == "-LOSS-":
        window['-MODEL-'].update(disabled=False)
        window['-MODEL_BROWSE-'].update(disabled=False)

window.close()
