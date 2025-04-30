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

@app.post("/segment")
async def segmentImage(
    file: UploadFile = File(...),
    points: str = Form(None),
    labels: str = Form(None)
):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = ImageDraw.Draw(image)

    image_np = np.array(image)

    # nparr = np.frombuffer(contents, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_np)
    h, w, _ = image_np.shape

    radius = 5
    x = w // 2
    y = h // 2
    print(f"heights: {h}, width: {w}")

    if points and labels:
        points = np.array(json.loads(points))
        # x1, y1 = points[0]
        x = points[0][0] * 3
        y = points[0][1] * 3
        # print(x1, y1)

    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
    image.save("image.png", format="PNG")

    input_points = np.array([[x, y]])
    input_labels = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    for i, (_, score) in enumerate(zip(masks, scores)):
        print(f"Mask No: {i}, Score: {score}, ")
    
    # print(masks[0])
    mask = masks[0].astype(np.uint8) * 255
    
    # 4. Convert mask to RGBA overlay
    mask_image = Image.fromarray(mask).convert("L")  # L mode = grayscale
    color_mask = Image.new("RGBA", (w, h), (0, 255, 0, 100))  # Semi-transparent green

    # Paste color only where mask exists
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.paste(color_mask, mask=mask_image)

    # 5. Convert original to RGBA
    image_rgba = image.convert("RGBA")

    # 6. Composite original image with mask overlay
    composite = Image.alpha_composite(image_rgba, overlay)
    composite.save("composite_image.png", format="PNG")


    # 7. Stream result back to frontend
    # buf = io.BytesIO()
    # composite.save(buf, format="PNG")
    # buf.seek(0)
    # print(composite)
    # print(image)

    # return StreamingResponse(buf, media_type="image/png")
    # return Response(content=composite, media_type="images/png")
    return { "message": "ok" }
