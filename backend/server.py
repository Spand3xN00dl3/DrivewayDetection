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
        responses={00: {"content": {"image/png": {}}}},
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

    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )

    best_i = 0

    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"Mask No: {i}, Score: {score}, ")

        if scores[i] > scores[best_i]:
            best_i = i
        # mask = mask.astype(np.uint8) * 255
        # create_composite_image(image, mask, f"composite{i}.png")
    
    
    buf = io.BytesIO()
    mask = masks[best_i].astype(np.uint8) * 255
    Image.fromarray(mask).convert("L").save(buf, format="PNG")
    # composite_image = create_composite_image(image, mask)
    # composite_image.save("images/composite_image.png", format="PNG")
    # composite_image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
    # return json.


@app.post("/calculateArea")
async def calculateArea(mask: UploadFile = File(...), image: UploadFile = File(...)):
    contents = await mask.read()
    mask = np.array(Image.open(io.BytesIO(contents)))
    contents = await image.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")



def create_composite_image(image, mask):
    h, w = mask.shape
    mask_image = Image.fromarray(mask).convert("L")  # L mode = grayscale
    color_mask = Image.new("RGBA", (w, h), (0, 255, 0, 100))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.paste(color_mask, mask=mask_image)
    image_rgba = image.convert("RGBA")
    composite = Image.alpha_composite(image_rgba, overlay)
    return composite


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
