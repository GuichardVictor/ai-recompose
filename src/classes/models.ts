import { Tensor, InferenceSession } from "onnxruntime-web";

class SAMMaskDecoder {

  model: InferenceSession
  imageEncoderAPIUrl: string

  constructor(imageEncoderAPIUrl: string) {
    this.imageEncoderAPIUrl = imageEncoderAPIUrl
  }

  load = async (modelPath: string) => {
    this.model = await InferenceSession.create(modelPath)
  }

  maskToImage = async (output) => {
    const [input, width, height] = [output.data, output.dims[2], output.dims[3]]

    const [r, g, b, a] = [0, 114, 189, 255];
    const arr = new Uint8ClampedArray(4 * width * height).fill(0);
    for (let i = 0; i < input.length; i++) {
      if (input[i] > 0.0) {
        arr[4 * i + 0] = r;
        arr[4 * i + 1] = g;
        arr[4 * i + 2] = b;
        arr[4 * i + 3] = a;
      }
    }
    const imageData = new ImageData(arr, height, width);
    return imageData;
  }

  blendMask = (image: ImageData, mask: ImageData): ImageData => {
    const { width, height, data } = image;

    for (let i = 0; i < width * height * 4; i += 4) {
      const alpha = 0.3; // Normalize alpha value to range [0, 1]

      // Blend each channel (R, G, B) using the canvas alpha
      data[i] = mask.data[i] * alpha + data[i] * (1 - alpha); // Red channel
      data[i + 1] = mask.data[i + 1] * alpha + data[i + 1] * (1 - alpha); // Green channel
      data[i + 2] = mask.data[i + 2] * alpha + data[i + 2] * (1 - alpha); // Blue channel
    }

    return image
  }

  getEmbeddings = async (image: File): Promise<Tensor> => {
    let data = new FormData()

    data.append('file', image)
    const options = {
      method: 'POST',
      body: data,
    };

    const response = await fetch(this.imageEncoderAPIUrl, options).then((e) => e.json())

    return new Tensor("float32", response.data, response.shape)
  }

  async predict(image: HTMLImageElement, embeddings: Tensor, clicks: number[][]): Promise<ImageData> {
    if (!image || !embeddings || !this.model)
      return

    let pointCoords = new Float32Array(2 * (clicks.length + 1));
    let pointLabels = new Float32Array(clicks.length + 1);

    let scale = 1024 / Math.max(image.height, image.width)

    for (let i = 0; i < clicks.length; i++) {
      pointCoords[2 * i] = clicks[i][0] * scale;
      pointCoords[2 * i + 1] = clicks[i][1] * scale;
      pointLabels[i] = 1;
    }

    pointCoords[2 * clicks.length] = 0.0;
    pointCoords[2 * clicks.length + 1] = 0.0;
    pointLabels[clicks.length] = -1.0;

    const pointCoordsTensor = new Tensor("float32", pointCoords, [1, clicks.length + 1, 2]);
    const pointLabelsTensor = new Tensor("float32", pointLabels, [1, clicks.length + 1]);

    const maskInput = new Tensor(
      "float32",
      new Float32Array(256 * 256),
      [1, 1, 256, 256]
    );

    // There is no previous mask, so default to 0
    const hasMaskInput = new Tensor("float32", [0]);
    const imageSizeTensor = new Tensor("float32", [
      image.height,
      image.width,
    ]);

    const feeds = {
      image_embeddings: embeddings,
      point_coords: pointCoordsTensor,
      point_labels: pointLabelsTensor,
      orig_im_size: imageSizeTensor,
      mask_input: maskInput,
      has_mask_input: hasMaskInput,
    }

    const results = await this.model.run(feeds);
    const output = results[this.model.outputNames[0]];

    const mask = await this.maskToImage(output)

    return mask
  }
}

export { SAMMaskDecoder }