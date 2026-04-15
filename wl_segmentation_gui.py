from tkinter import (
    Button, Entry, Frame, Label, StringVar,
    E, W, Tk, filedialog, ttk
)

import numpy as np

from gradcam import GradCAM, show_gradcam, get_target_layer
from model import set_device, load_model

MODEL_PATH = "model/unet_20260328_211654_epoch37_iou0.742.pth"


class Window(Frame):

    def __init__(self, master=None):
        ###################################
        # This is where we create our GUI #
        ###################################

        Frame.__init__(self, master)
        self.master = master

        # Input image
        img_frame = Frame(master, padx=10, pady=5)  # Defines the widget group
        img_label = Label(img_frame, text="Input image file name:")
        img_label.grid(column=0, row=0, sticky=W)
        self.img_file_name = StringVar()
        img_entry = Entry(img_frame, width=40, textvariable=self.img_file_name)
        img_entry.grid(column=1, row=0, sticky=E)
        img_button = Button(img_frame, text="...", command=self.img_file_selector)
        img_button.grid(column=2, row=0)
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid(column=0, row=0, sticky=W + E)  # First row of widgets

        # Ground truth
        gt_frame = Frame(master, padx=10, pady=5)  # Defines the widget group
        gt_label = Label(gt_frame, text="Ground truth image file name:")
        gt_label.grid(column=0, row=0, sticky=W)
        self.gt_file_name = StringVar()
        gt_entry = Entry(gt_frame, width=40, textvariable=self.gt_file_name)
        gt_entry.grid(column=1, row=0, sticky=E)
        gt_button = Button(gt_frame, text="...", command=self.gt_file_selector)
        gt_button.grid(column=2, row=0)
        gt_frame.grid_columnconfigure(1, weight=1)
        gt_frame.grid(column=0, row=1, sticky=W + E)  # Second row of widgets

        # Grad-CAM target layer
        layer_frame = Frame(master, padx=10, pady=5)
        layer_label = Label(layer_frame, text="Grad-CAM target layer:")
        layer_label.grid(column=0, row=0, sticky=W)
        self.layer_var = StringVar()
        layer_options = [
            "down1", "down2", "down3", "down4", "bottleneck", "conv4", "conv3", "conv2", "conv1"
        ]
        layer_dropdown = ttk.Combobox(
            layer_frame,
            textvariable=self.layer_var,
            values=layer_options,
            state="readonly",  # user cannot type, only select
            width=37
        )
        layer_dropdown.grid(column=1, row=0, sticky=E)
        layer_dropdown.current(0)  # default value
        layer_frame.grid_columnconfigure(1, weight=1)
        layer_frame.grid(column=0, row=2, sticky=W + E)

        # Output file name
        output_frame = Frame(master, padx=10, pady=5)  # Defines the widget group
        output_label = Label(output_frame, text="Output file name:")
        output_label.grid(column=0, row=0, sticky=W)
        self.output_file_name = StringVar()
        output_entry = Entry(output_frame, width=40, textvariable=self.output_file_name)
        output_entry.grid(column=1, row=0, sticky=E)
        output_button = Button(output_frame, text="...", command=self.output_file_selector)
        output_button.grid(column=2, row=0)
        output_frame.grid_columnconfigure(1, weight=1)
        output_frame.grid(column=0, row=3, sticky=W + E)  # Fourth row of widgets

        bottom_frame = Frame(master, padx=10, pady=10)

        # Exit and OK buttons
        btn_frame = Frame(bottom_frame)
        ok_btn = Button(btn_frame, text="OK", command=self.run_segmentation)
        ok_btn.grid(column=0, row=0, sticky=W)
        exit_btn = Button(btn_frame, text='Exit', command=self.exit)
        exit_btn.grid(column=1, row=0, sticky=W)
        btn_frame.grid(column=0, row=4, sticky=W)  # Fifth row of widgets

        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid(column=0, row=4, sticky=W + E)

    # The method in which we run our model. This is called when we press the 'OK' button.
    def run_segmentation(self):

        try:
            # load files
            img = np.load(self.img_file_name.get())
            mask = np.load(self.gt_file_name.get())

            grad_cam = GradCAM(model, get_target_layer(model, self.layer_var.get()))
            show_gradcam(model, img, mask, grad_cam, self.output_file_name.get())

        except Exception as e:
            print("The error raised is: ", e)
        finally:
            print("All Done!")
            self.exit()

    def exit(self):
        exit(0)

    def img_file_selector(self):
        fn = filedialog.askopenfilename(title='Select the input image file')
        self.img_file_name.set(fn)

    def gt_file_selector(self):
        fn = filedialog.askopenfilename(title='Select the groundtruth file')
        self.gt_file_name.set(fn)

    def output_file_selector(self):
        fn = filedialog.asksaveasfilename(title='Save As')
        self.output_file_name.set(fn)

device = set_device()
model = load_model(device, MODEL_PATH)

root = Tk()
app = Window(root)
root.wm_title("Plot Timeseries Data")
root.mainloop()