from typing import Annotated
import io

import uvicorn
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np

app = FastAPI()

checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

cache = {}


@app.post("/predict/embeddings")
async def predict(file: Annotated[bytes, File()]):
    image = np.array(Image.open(io.BytesIO(file)))
    if file not in cache:
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        cache[file] = image_embedding
    
    image_embedding = cache[file]

    return {"data": image_embedding.reshape(-1).tolist(), "shape": image_embedding.shape}



@app.get("/")
async def root():
    return {"message": "Hello World"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)