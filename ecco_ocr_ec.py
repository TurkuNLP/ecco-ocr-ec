import torch
import transformers
import argparse
import datasets
import deepspeed
import pathlib
import csv
import os
import pytorch_lightning as pl

# torch.use_cache = False

# model_name = 'EleutherAI/gpt-neo-2.7B'
model_name = 'EleutherAI/gpt-j-6B'
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
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.lr = lr
        self.steps_train = steps_train

    # def configure_sharded_model(self):

    def forward(self, batch):
        return self.model(**batch)

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
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps=self.steps_train)
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
        return torch.utils.data.DataLoader(torch_dataset, collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False), batch_size=self.batch_size, sampler=sampler, num_workers=1, pin_memory=True)

    def val_dataloader(self):
        torch_dataset = OCRDataSet(self.dataset['test'])
        sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset, shuffle=False)
        return torch.utils.data.DataLoader(torch_dataset, collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False), batch_size=self.batch_size, sampler=sampler, num_workers=1, pin_memory=True)

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=1, help="Number of nodes.")
parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use per node.")
parser.add_argument('--train', help="A directory with csv files, each having a header line, with each row containing a correct page and its OCR equivalent")
parser.add_argument('--out_dir', help="A directory to which the model checkpoints are saved.")
parser.add_argument('--load_checkpoint', help="A path to a checkpoint file to load.")
args = parser.parse_args()

accumulate_grad_batches = 8
steps_train = 80000
lr = 2e-5
local_batch_size = 1
max_length = 1536

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")
print(f"Number of nodes: {args.nodes}, number of GPUs per node: {args.gpus}")
print(f"Learning rate: {2e-5}, local batch size: {local_batch_size}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")

train_fns = [p for p in pathlib.Path(args.train).iterdir() if p.is_file()][:100]

page_pairs = []
for fn in train_fns:
    with open(fn, 'rt', newline='') as f:
        reader = csv.reader(f)
        next(reader) # Skip the header line.
        page_pairs += [{'ocr': row[1], 'tcp': row[2]} for row in reader]

print(f"Read {len(page_pairs)} page pairs.")
dataset = datasets.Dataset.from_list(page_pairs)
dataset = dataset.train_test_split(test_size=0.1)
print(f"Average OCR page length in test: {sum([len(d['ocr']) for d in page_pairs])/len(page_pairs)}")
print(f"Average clean page length in test: {sum([len(d['tcp']) for d in page_pairs])/len(page_pairs)}")

# https://huggingface.co/docs/transformers/tasks/language_modeling
# TODO: Replace with code that doesn't result in OOM when attempting to tokenize the entire dataset.
dataset = dataset.map(
    lambda b: tokenizer([o+'\n<OCR>\n'+t for o, t in zip(b['ocr'], b['tcp'])], max_length=max_length, truncation=True),
    batched=True,
    num_proc=4
)

dataset = dataset.remove_columns(['ocr', 'tcp'])

print(dataset)

truncated = sum(len(e) == max_length for e in dataset['train']['input_ids'] + dataset['test']['input_ids'])
print(f"Number of maximum length samples: {truncated}, proportion: {truncated / (len(dataset['train']) + len(dataset['test']))}")

datamodule = OCRDataModule(dataset, tokenizer, local_batch_size)
gpt_model = GPTModel(model_name, lr, steps_train)

checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=1000, monitor='global_step', mode='max', save_top_k=-1, dirpath=args.out_dir, filename='{global_step}')

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
    strategy='deepspeed_stage_3_offload',
    # precision=16,
    # gradient_clip_algorithm='norm',
    # gradient_clip_val=1.0,
    accumulate_grad_batches=accumulate_grad_batches,
    val_check_interval=100,
    limit_val_batches=100,
    # max_epochs=1,
    max_steps=steps_train,
    callbacks=[checkpoint_callback, pl.callbacks.TQDMProgressBar(refresh_rate=10)]
)

print(trainer.global_rank, trainer.world_size, os.environ['SLURM_NTASKS'])

trainer.fit(gpt_model, datamodule=datamodule)
