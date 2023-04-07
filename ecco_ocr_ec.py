import torch
import transformers
import datasets
import evaluate
import deepspeed
import argparse
import pathlib
import csv
import os
import logging
import math
import pytorch_lightning as pl

# torch.use_cache = False

model_name = 'gpt2'
# model_name = 'EleutherAI/gpt-neo-2.7B'
# model_name = 'EleutherAI/gpt-j-6B'
# model_name = 'EleutherAI/gpt-neox-20b'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Note: if the model is moved to GPU here by using .cuda(), all of the processes in the node end up
# using the same GPU, instead of each using a different GPU.
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

# print(model)

class GPTModel(pl.LightningModule):
    def __init__(self, model_name, lr, steps_train):
        super().__init__()
        self.save_hyperparameters()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.lr = lr
        self.steps_train = steps_train

    # def configure_sharded_model(self):

    def forward(self, batch):
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          labels=batch['labels'])

    def training_step(self, batch):
        out = self(batch)
        self.log_dict({'loss': out.loss, 'global_step': self.trainer.global_step}, sync_dist=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        # if batch_idx % 100 == 0:
        #     print(batch, flush=True)
        self.log('val_loss', self(batch).loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Instead of self.parameters(), self.trainer.model.parameters() must be used with FSDP auto-wrapping.
        # See: https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#auto-wrapping
        # optimizer = transformers.optimization.AdamW(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.01)
        optimizer = deepspeed.ops.adam.FusedAdam(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

class OCRDataSet(torch.utils.data.Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

class PromptMaskingDataCollator(transformers.DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        data = super().__call__(features, return_tensors)

        for i, prefix_len in enumerate(data['prefix_length']):
            data['labels'][i,:prefix_len] = -100
            # Since the EOS token is used as the PAD token, EOS is masked and must be added back.
            data['labels'][i, torch.nonzero(data['input_ids'][i] == tokenizer.eos_token_id)[0]] = tokenizer.eos_token_id

        return data

class OCRDataModule(pl.LightningDataModule):
    def __init__(self, dataset, tokenizer, batch_size):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def train_dataloader(self):
        torch_dataset = OCRDataSet(self.dataset['train'])
        # With DDP, PyTorch Lightning wraps the Dataloader with DistributedSampler automatically.
        # With FSDP, it seems this has to be done manually.
        sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset, shuffle=True)
        return torch.utils.data.DataLoader(torch_dataset, collate_fn=PromptMaskingDataCollator(tokenizer=self.tokenizer, mlm=False), batch_size=self.batch_size, sampler=sampler, num_workers=1, pin_memory=True)

    def val_dataloader(self):
        torch_dataset = OCRDataSet(self.dataset['test'])
        sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset, shuffle=False)
        return torch.utils.data.DataLoader(torch_dataset, collate_fn=PromptMaskingDataCollator(tokenizer=self.tokenizer, mlm=False), batch_size=self.batch_size, sampler=sampler, num_workers=1, pin_memory=True)

def filter_by_length(datasetdict, max_length):
    for k in datasetdict:
        filtered = datasetdict[k].filter(lambda e: len(e['input_ids']) <= max_length)
        orig_length = len(datasetdict[k]['input_ids'])
        filt_length = len(filtered['input_ids'])
        print(f'filtered {k} from {orig_length} to {filt_length}')
        print(f'({filt_length/orig_length:.1%}) by max_length {max_length}')
        datasetdict[k] = filtered

    return datasetdict

def compute_metrics(predictions, references):
    cer = evaluate.load('character').compute(predictions=predictions, references=references)
    wer = evaluate.load('wer').compute(predictions=predictions, references=references)
    return {'cer': cer['cer_score'], 'wer': wer}

def tokenize_with_prefix_length(tokenizer, b):
    prefix = tokenizer([i+'\nCorrect:\n' for i in b['input']], truncation=False)
    output = tokenizer([o + tokenizer.eos_token for o in  b['output']], truncation=False)
    d = {**{k: [p + o for p, o in zip(prefix[k], output[k])] for k in prefix}, 'prefix_length': [len(p) for p in prefix['input_ids']]}
    return d

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=1, help="Number of nodes.")
parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use per node.")
parser.add_argument('--train', help="A jsonl file, with each row containing a noisy text 'input' and its correct form 'output'.")
parser.add_argument('--eval', help="A jsonl file in the same format as the --train argument.")
parser.add_argument('--out_dir', help="A directory to which the model checkpoints are saved.")
parser.add_argument('--load_checkpoint', help="A path to a checkpoint file to load.")
args = parser.parse_args()

accumulate_grad_batches = 1
# steps_train = 80000
lr = 5e-5
local_batch_size = 8
max_length = 512
eval_size = 1000

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")
print(f"Number of nodes: {args.nodes}, number of GPUs per node: {args.gpus}")
print(f"Learning rate: {2e-5}, local batch size: {local_batch_size}, maximum sequence length: {max_length}")
# print(f"Number of training steps: {steps_train}")
print(f"Number of evaluation examples: {eval_size}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")

# train_fns = [p for p in pathlib.Path(args.train).iterdir() if p.is_file()]

# page_pairs = []
# for fn in train_fns:
#     with open(fn, 'rt', newline='') as f:
#         reader = csv.reader(f)
#         next(reader) # Skip the header line.
#         page_pairs += [{'ocr': row[1], 'tcp': row[2]} for row in reader]
 

# print(f"Read {len(page_pairs)} page pairs.")
# dataset = datasets.Dataset.from_list(page_pairs)
# dataset = dataset.train_test_split(test_size=0.1)
# print(f"Average OCR page length in test: {sum([len(d['ocr']) for d in page_pairs])/len(page_pairs)}")
# print(f"Average clean page length in test: {sum([len(d['tcp']) for d in page_pairs])/len(page_pairs)}")

dataset = datasets.load_dataset('json', data_files={'train': args.train, 'test': args.eval})
dataset['test'] = dataset['test'].select(range(eval_size))
# dataset['train'] = dataset['train'].select(range(100000))
print(dataset['test'][0])
print(dataset['test'][-1])
# print(list(zip(dataset['test']['input'][:10], dataset['test']['output'][:10])))
print(compute_metrics(predictions=dataset['test']['input'], references=dataset['test']['output']))

# https://huggingface.co/docs/transformers/tasks/language_modeling
# TODO: Replace with code that doesn't result in OOM when attempting to tokenize the entire dataset.
dataset = dataset.map(
    lambda b: tokenize_with_prefix_length(tokenizer, b),
    batched=True,
    num_proc=4
)

# metrics = evaluate.combine(['character', 'wer'], force_prefix=True)
# print(metrics.compute(predictions=dataset['test']['input'], references=dataset['test']['output']))

# print(dataset['train'][:10])
# print(dataset['test'][:10])
dataset = dataset.remove_columns(['input', 'output'])
# Inputs longer than the maximum length of the model are removed from the dataset.
dataset = filter_by_length(dataset, max_length)
print(dataset)

# truncated = sum(len(e) == max_length for e in dataset['train']['input_ids'] + dataset['test']['input_ids'])
# print(f"Number of maximum length samples: {truncated}, proportion: {truncated / (len(dataset['train']) + len(dataset['test']))}")

datamodule = OCRDataModule(dataset, tokenizer, local_batch_size)
steps_train = math.ceil(len(dataset['train']) / (args.gpus*local_batch_size))
print(f"Number of training steps: {steps_train}", flush=True)

if args.load_checkpoint:
    gpt_model = GPTModel.load_from_checkpoint(args.load_checkpoint)
    print(f"Model loaded from checkpoint: {args.load_checkpoint}")
else:
    gpt_model = GPTModel(model_name, lr, steps_train)

# checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=1000, monitor='global_step', mode='max', save_top_k=-1, dirpath=args.out_dir, filename='{global_step}')
checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=1, dirpath=args.out_dir, filename=model_name+'-{epoch}')

# fsdp = pl.strategies.DDPFullyShardedNativeStrategy(
#     cpu_offload=torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload(offload_params=True),
#     activation_checkpointing=transformers.models.gptj.modeling_gptj.GPTJBlock
#     # activation_checkpointing=transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock
# )

trainer = pl.Trainer(
    num_nodes=args.nodes,
    accelerator='gpu',
    devices=args.gpus,
    # auto_select_gpus=True,
    # strategy=fsdp,
    strategy='deepspeed_stage_2',
    # precision=16,
    # gradient_clip_algorithm='norm',
    # gradient_clip_val=1.0,
    accumulate_grad_batches=accumulate_grad_batches,
    val_check_interval=500,
    # limit_val_batches=100,
    max_epochs=1,
    # max_steps=steps_train,
    callbacks=[checkpoint_callback, pl.callbacks.TQDMProgressBar(refresh_rate=10)],
    resume_from_checkpoint=args.load_checkpoint
)

print(trainer.global_rank, trainer.world_size, os.environ['SLURM_NTASKS'])

trainer.fit(gpt_model, datamodule=datamodule)

gpt_model.eval()
gpt_model.cuda()
if trainer.global_rank == 0:
    predictions = []
    references = []
    for cpu_batch in datamodule.val_dataloader():
        batch = {k: v.cuda() for k, v in cpu_batch.items()}
        # print([(p.shape, p[:l].shape) for p, l in zip(batch['input_ids'], batch['prefix_length'])])
        output = [gpt_model.model.generate(torch.unsqueeze(p[:l], 0), do_sample=False, max_length=max_length).squeeze() for p, l in zip(batch['input_ids'], batch['prefix_length'])]
        # print(output)
        # Predictions might not have an EOS token. In these cases, model output is not truncated.
        # for o, l in zip(output, batch['prefix_length']):
        #     print([torch.nonzero(o == tokenizer.eos_token_id), torch.tensor([[max_length]])])
        max_value = torch.tensor([[max_length]]).cuda()
        predictions += [o[l:torch.concat([torch.nonzero(o == tokenizer.eos_token_id), max_value])[0]] for o, l in zip(output, batch['prefix_length'])]
        references += [o[l:torch.nonzero(o == tokenizer.eos_token_id)[0]] for o, l in zip(batch['input_ids'], batch['prefix_length'])]

    # print(predictions)
    # print(references)
    predictions = tokenizer.batch_decode(predictions)    
    references = tokenizer.batch_decode(references)

    print(list(zip(predictions, references))[:10])
    print(compute_metrics(predictions=predictions, references=references))
