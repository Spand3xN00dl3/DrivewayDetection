from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import numpy as np
import io
import json
import torch
import torchvision
import cv2
from segment_anything import sam_model_registry, SamPredictor

from dotenv import load_dotenv
import os
import requests




load_dotenv()
latest_mask = []


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

@app.get("/")
async def root():
    return { "message": "hello world"}

@app.post(
        "/segment",
        response={00: {"content": {"image/png": {}}}},
        response_class=Response
)
async def segmentImage(
    file: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_np = np.array(image)

    predictor.set_image(image_np)
    h, w, _ = image_np.shape

    radius = 5
    points = np.array(json.loads(points))
    labels = np.array(json.loads(labels))
    print(f"Points: {points}, Labels: {labels}")

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="blue" if labels[i] else "red")
    
    # save_image(image, "points.png")

#     if points:
#         return {"message": "ok"}

#     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
#     # save_image(image, "image.png")

#     input_points = np.array([[x, y]])
#     input_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"Mask No: {i}, Score: {score}, ")
        mask = mask.astype(np.uint8) * 255
        # print(mask)
        create_composite(image, mask, f"composite{i}.png")
    
#     latest_mask = mask
#     print("latest mask")
#     print(latest_mask)

#     return { "message": "ok" }

def create_composite(image, mask, filename="composite_image.png"):
    # 4. Convert mask to RGBA overlay
    h, w = mask.shape
    mask_image = Image.fromarray(mask).convert("L")  # L mode = grayscale
    color_mask = Image.new("RGBA", (w, h), (0, 255, 0, 100))  # Semi-transparent green

    # Paste color only where mask exists
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.paste(color_mask, mask=mask_image)

    # 5. Convert original to RGBA
    image_rgba = image.convert("RGBA")

    # 6. Composite original image with mask overlay
    composite = Image.alpha_composite(image_rgba, overlay)
    save_image(composite, filename)


@app.post("/locate")
async def locateDriveway(
    address: str
):
    response = requests.post()
    return {"message": "ok"}


def save_image(image, filename, format="PNG"):
    try:
        image.save(f"images/{filename}", format=format)
        return True
    
    except:
        return False


def shoelace_formula(points):
    if len(points) <= 2:
        return 0

    a1 = 0
    a2 = 0

    for i in range(len(points) - 1):
        a1 += points[i][0] * points[i + 1][1]
        a2 += points[i][1] * points[i + 1][0]
    
    a1 += points[-1][0] * points[0][1]
    a2 += points[-1][1] * points[0][0]

    return abs(a1 + a2) / 2.0


# print("test")
# points = [[0, 0], [1, 0], [1, 2], [0, 0]]
# print(shoelace_formula(points))
import uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
