
<h1 align="center">
INFERCEPT: Efficient Intercept Support for Large-Language Model
Inferencing
</h1>

<!-- <p align="center"> -->
<!-- | <a href=""><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://discord.gg/jz7wjKhh6g"><b>Discord</b></a> | -->
<!-- </p> -->

This repo contains implementation of InferCept. Please refer to our <a href='https://arxiv.org/abs/2402.01869'><b>paper</b></a> for more details.
---
## Instructions
To install InferCept to your environment:
```bash
# After cloning the repo
cd infercept/
pip install -e .
```

To enable the serving system to hook on augmentation calls, register your aug-stop token in `vllm/utils.py`. You can register multiple keys at once:

```python
def get_api_stop_strings() -> List[str]:
  return ["<stop token 1>", "<stop token 2>"]
```

To reproduce paper results, check `exps` folder.
## Citation

If you use InferCept for your research, please cite our paper:
```bibtex
@inproceedings{
  abhyankar2024infer,
  title={INFERCEPT: Efficient Intercept Support for Large-Language Model
Inferencing},
  author={Reyna Abhyankar and Zijian He and Vikranth Srivatsa and Hao Zhang and Yiying Zhang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  month=Jul,
  address={Vienna, Austria},
}
```
