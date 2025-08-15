import os
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import test_collate_function, visualize_bounding_boxes

# small GUI bits
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, Image

os.makedirs("outputs", exist_ok=True)

# ---------- CONFIG ----------
# If True => show ROI selector and process only inside ROI.
# If False => same as previous behavior (process full image).
crop_to_drawn_area = True
# ----------------------------

def get_dataloader(processor):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader

# Minimal Tkinter ROI selector: returns (x1,y1,x2,y2) or None
def select_roi_tk(pil_image, title="Select ROI", instr="Draw a green rectangle with the mouse, then close this window."):
    state = {"drawing": False, "start": (-1, -1), "end": (-1, -1)}

    def on_down(ev):
        state["drawing"] = True
        state["start"] = (ev.x, ev.y)

    def on_move(ev):
        if state["drawing"]:
            canvas.delete("temp_rect")
            sx, sy = state["start"]
            canvas.create_rectangle(sx, sy, ev.x, ev.y, outline="green", width=3, tags="temp_rect")

    def on_up(ev):
        state["drawing"] = False
        state["end"] = (ev.x, ev.y)

    def on_close():
        root.quit()

    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=instr, fg="blue", font=("Tahoma", 11)).pack()
    canvas = tk.Canvas(root, width=pil_image.width, height=pil_image.height, highlightthickness=0)
    canvas.pack()

    # Show image (optionally with a small translucent hint)
    display_img = pil_image.convert("RGBA").copy()
    draw_temp = ImageDraw.Draw(display_img)
    hint_text = "Draw a green rectangle"
    bbox = draw_temp.textbbox((0, 0), hint_text)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    padding = 8
    draw_temp.rectangle((10, 10, 10 + tw + padding, 10 + th + padding // 2), fill=(255,255,255,200))
    draw_temp.text((14, 12), hint_text, fill="black")

    photo = ImageTk.PhotoImage(display_img)
    canvas.create_image(0, 0, image=photo, anchor="nw")

    canvas.bind("<Button-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
    root.destroy()

    sx, sy = state["start"]
    ex, ey = state["end"]
    if sx == -1 or ex == -1:
        return None

    x1, y1 = max(0, min(sx, ex)), max(0, min(sy, ey))
    x2, y2 = min(pil_image.width, max(sx, ex)), min(pil_image.height, max(sy, ey))
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None

if __name__ == "__main__":
    cfg = Configuration()
    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
    )
    model.eval()
    model.to(cfg.device)

    test_dataloader = get_dataloader(processor=processor)

    # get a single batch like original code (but we'll prepare inputs per-image below)
    sample_batch = next(iter(test_dataloader))
    # depending on your collate, sample_batch may be (sample, sample_images)
    # try to unpack safely:
    try:
        sample_tensor_batch, sample_images = sample_batch
    except Exception:
        # fallback: assume the second element is images
        sample_images = sample_batch[1]

    file_count = 0
    for sample_image in sample_images:
        # sample_image in your original code was indexed [0] to get PIL image
        if isinstance(sample_image, (list, tuple)):
            original_image = sample_image[0]
        else:
            original_image = sample_image

        if not isinstance(original_image, Image.Image):
            raise ValueError("Expected PIL.Image in sample_images; got: " + str(type(original_image)))

        # decide ROI
        roi_coords = None
        if crop_to_drawn_area:
            print("Please draw ROI for this image window. If you close without drawing, the full image will be used.")
            roi = select_roi_tk(original_image,
                                title="Select ROI",
                                instr="Draw a green rectangle with the mouse, then close this window.")
            if roi is not None:
                roi_coords = roi
                roi_image = original_image.crop(roi_coords)
                print(f"Using ROI {roi_coords}, size: {roi_image.size}")
            else:
                roi_image = original_image
                print("No ROI drawn; using full image.")
        else:
            roi_image = original_image

        # Prepare model inputs for this (roi_image or full image)
        inputs = processor(images=roi_image, return_tensors="pt")
        # move tensors to device
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

        # generate and decode
        generation = model.generate(**inputs, max_new_tokens=100)
        decoded = processor.batch_decode(generation, skip_special_tokens=True)
        output_text = decoded[0]
        print(f"Model output for file {file_count}: {output_text}")

        # Save results:
        # If ROI was used: annotate ROI (using your visualize_bounding_boxes), paste back to original, draw green rect
        if roi_coords is not None:
            tmp_path = f"outputs/_tmp_roi_{file_count}.png"
            # visualize_bounding_boxes signature in your original code: (image, output_text, width, height, out_path)
            visualize_bounding_boxes(roi_image.copy(), output_text, roi_image.width, roi_image.height, tmp_path)
            annotated_roi = Image.open(tmp_path).convert("RGB")
            base = original_image.convert("RGB").copy()
            base.paste(annotated_roi, (roi_coords[0], roi_coords[1]))
            draw = ImageDraw.Draw(base)
            draw.rectangle(roi_coords, outline="green", width=4)
            out_path = f"outputs/output_{file_count}.png"
            base.save(out_path)
            # cleanup temp
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # full image processing as before
            out_path = f"outputs/output_{file_count}.png"
            visualize_bounding_boxes(original_image.copy(), output_text, original_image.width, original_image.height, out_path)

        print(f"Saved: {out_path}")
        file_count += 1

    print("Done.")
