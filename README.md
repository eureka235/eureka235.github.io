# Speech World Model

<img src="resources/pipeline.png" alt="" style="zoom: 20%; display: block; margin-right: auto; margin-left: 0;" />


## Vicuna Deployment

```sh
export CUDA_VISIBLE_DEVICES=4,5

python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path /data/xxx/vicuna-13b-v1.5
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

## Environment Configuration
Please refer [requirements.txt](requirements.txt)


## Casual Graph
