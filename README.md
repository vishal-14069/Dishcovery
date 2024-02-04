## Food Classification App using Vision Transformer (WIP)

The Food Classification App is powered by a Pretrained <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a> Model

  The app predicts the food in the given image, and provides recipies and information about the food

  Vision Transformer classifies the image -----> Text Generation Model provides recipies and information about the food


  ViT's  classifier layer is fine-tuned on ---> https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset/data


  To get started with training & inference:

  ```sh
  cd src
  ```

  ```sh
  python train.py --num_epochs 10 \
                  --batch_size 32 \
                  --learning_rate 1e-3 \
                  --train_dir "./ViT-using-PyTorch/data/train" \
                  --test_dir "./ViT-using-PyTorch/data/test" \
                  --device "cpu" \
  ```

Run Inference:

```sh
  python predict.py --image_path "pizza.jpg" --device "cpu"
```

To Do:

* Train the model on a larger dataset
* Quantize the model for faster inference and deployment
* Work on the text generation model
* Build a Streamlit/Gradio UI
* Deploy!



  

