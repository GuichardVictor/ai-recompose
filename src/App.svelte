<script lang="ts">
  import { onMount } from "svelte";
  import { Tensor, InferenceSession } from "onnxruntime-web";
  import { SAMMaskDecoder } from "./classes/models";
  import { debounce } from "./classes/utils";

  // UI
  let canvas;
  let imageInput;
  let embeddingPromise;
  let mask;

  // ORT Inputs
  let imageData;
  let embeddings;
  let clicks = [];
  let model: SAMMaskDecoder;

  const handleMouseMove = (canvas, e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    clicks = [[x, y]];
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    let reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onload = (e) => {
      let image = new Image();
      image.src = e.target.result;

      image.onload = async (e) => {
        const drawableRatio = image.width / image.height;
        let canvasContext = canvas.getContext("2d");

        image.width = Math.min(1280, image.width);
        image.height = parseInt(image.width / drawableRatio, 10);

        canvas.width = image.width;
        canvas.height = image.height;

        imageData = image;

        canvasContext.drawImage(image, 0, 0, image.width, image.height);

        embeddingPromise = model.getEmbeddings(file);
        embeddings = await embeddingPromise;
      };
    };
  };

  $: if (clicks.length > 0) {
    const samPredict = async () => {
      mask = await model.predict(imageData, embeddings, clicks);
    };

    samPredict();
  }

  $: if (mask && model.model) {
    const ctx = canvas.getContext("2d");
    ctx?.drawImage(imageData, 0, 0, imageData.width, imageData.height);
    let canvasImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    canvasImageData = model.blendMask(canvasImageData, mask);
    ctx.putImageData(canvasImageData, 0, 0);
  }

  onMount(() => {
    const loadModel = async () => {
      model = new SAMMaskDecoder("http://127.0.0.1:8000/predict/embeddings");
      model.load("/models/sam_onnx_quantized.onnx");
    };
    loadModel();

    canvas.addEventListener(
      "mousemove",
      debounce((e) => {
        handleMouseMove(canvas, e);
      }, 100)
    );

    canvas.addEventListener("mouseout", (e) => {
      mask = undefined;
      if (imageData) {
        const ctx = canvas.getContext("2d");
        ctx?.drawImage(imageData, 0, 0, imageData.width, imageData.height);
      }
    });
  });
</script>

<main>
  <input
    accept="image/png, image/jpeg"
    bind:this={imageInput}
    id="image"
    name="image"
    type="file"
    on:change={handleFileUpload}
  />

  {#if embeddingPromise}
    {#await embeddingPromise}
      <p>Loading...</p>
    {:then res}
      <p>Ready</p>
    {/await}
  {/if}

  <canvas
    style="border: solid;"
    bind:this={canvas}
    width="1200"
    height="720"
    color="red"
  />
</main>

<style>
</style>
