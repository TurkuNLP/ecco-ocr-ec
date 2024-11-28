import torch
import transformers
import deepspeed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds_checkpoint', help='Path to a DeepSpeed checkpoint directory.')
parser.add_argument('--model_path', help='Path to which the model is saved.')
args = parser.parse_args()

# state_dict = torch.load(args.ds_checkpoint)
# torch.save(state_dict, args.model_path)

state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(args.ds_checkpoint)
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
print([k for k in state_dict.keys()])
model = transformers.BloomForCausalLM.from_pretrained('LumiOpen/Poro-34B', state_dict=state_dict)
model.save_pretrained(args.model_path)
