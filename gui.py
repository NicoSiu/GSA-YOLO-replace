#import
from super_gradients.common.object_names import Models
from super_gradients.training import models
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
from gradio_app import run_grounded_sam
from grounded_sam_inpainting_demo import ground_sam_inpainting
from automatic_label_tag2text_demo import automatic_label_tag2text
from grounded_sam_whisper_demo import ground_sam_whisper


# For function selection
def task_selection(task_type, input_image, input_text, inpaint_text, input_audio, box_threshold, iou_threshold, text_threshold):
    if task_type == "YOLO-NAS":

        res_img = object_detection(input_image, box_threshold, iou_threshold)
        return res_img

    if task_type == "Grounded-SAM":

        res_img = Grounded_SAM(input_image, input_text, box_threshold, iou_threshold, text_threshold)
        return res_img

    if task_type == "Grounded-SAM with Inpainting":

        res_img = Grounded_SAM_with_Inpainting(input_image, input_text, inpaint_text, box_threshold, text_threshold)
        return res_img

    if task_type == "Grounded-SAM for Automatic Labeling":

        res_img = Grounded_SAM_for_Automatic_Labeling(input_image, box_threshold, iou_threshold, text_threshold)
        return res_img

    if task_type == "Grounded-SAM with Whisper":

        res_img = Grounded_SAM_with_Whisper(input_image, input_audio, box_threshold, iou_threshold, text_threshold)
        return res_img

    else:
        print("Please select one task!")

# YOLO-NAS
def object_detection(input_image, box_threshold, iou_threshold):

    # load image
    #print(torch.cuda.is_available())
    model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # detection
    image = input_image["image"]
    predictions = model.predict(image, iou_threshold, box_threshold)
    img2 = predictions[0].draw()
    img2 = Image.fromarray(img2)

    return img2

# Grounded_SAM
def Grounded_SAM(input_image, input_text, box_threshold, iou_threshold, text_threshold):

    [image_pil, mask_image] = run_grounded_sam(input_image, input_text, "seg", "", box_threshold, iou_threshold, text_threshold, "merge", "split", "")
    #size = image_pil.size
    #image_pil = image_pil.resize(size)

    return image_pil

# Grounded_SAM_with_inpainting
def Grounded_SAM_with_Inpainting(input_image, input_text, inpaint_text, box_threshold, text_threshold):

    img = ground_sam_inpainting(input_image, input_text, inpaint_text, box_threshold, text_threshold)
    return img

# Grounded_SAM_for_Automatic_Labeling
def Grounded_SAM_for_Automatic_Labeling(input_image, box_threshold, iou_threshold, text_threshold):

    img = automatic_label_tag2text(input_image, box_threshold, iou_threshold, text_threshold)

    return img

def Grounded_SAM_with_Whisper(input_image, input_audio, box_threshold, iou_threshold, text_threshold):

    img = ground_sam_whisper(input_image, input_audio, box_threshold, iou_threshold, text_threshold)

    return img

# Grounded gui design
def gui():
    app = gr.Blocks()
    with app: # Create a block for block
        with gr.Row():
            with gr.Column(): # First column

                task_type = gr.Dropdown(["YOLO-NAS", "Grounded-SAM", "Grounded-SAM with Inpainting", "Grounded-SAM for Automatic Labeling", "Grounded-SAM with Whisper"],
                                        label="Welcome to the GSA-YOLO demo, pls select your task below")
                input_image = gr.Image(label='upload image', source="upload", type="pil", tool="sketch")
                input_text = gr.Textbox(label="Object text")
                inpaint_text = gr.Textbox(label="Inpaint text")
                input_audio = gr.Audio(label='upload audio', source="upload", type="numpy")

                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05
                    )

                run_button = gr.Button(label="Run")

            with gr.Column():  # Second column
                output = gr.Image(label='Output image')

                # core function
                run_button.click(fn=task_selection, inputs=[task_type, input_image, input_text, inpaint_text, input_audio, box_threshold, iou_threshold, text_threshold], outputs=output)
                app.launch(server_name='0.0.0.0', server_port=1122, debug="store_true", share="store_true")


#main
if __name__ == "__main__":
    gui()