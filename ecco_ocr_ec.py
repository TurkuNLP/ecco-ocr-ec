import torch
import transformers
import datasets
import evaluate
import deepspeed
import argparse
import pathlib
import os
import math
import itertools
import time
import pytorch_lightning as pl
# from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding
import matplotlib.pyplot as plt
import seaborn as sns

# torch.use_cache = False

# model_name = 'gpt2'
# model_name = 'facebook/opt-1.3b'
# model_name = 'facebook/opt-13b'
# model_name = 'EleutherAI/gpt-neo-2.7B'
model_name = 'EleutherAI/gpt-j-6B'
# model_name = 'EleutherAI/gpt-neox-20b'
# model_name = 'EleutherAI/pythia-6.9b'
# model_name = 'EleutherAI/pythia-12b'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# print(f"Pad token: {tokenizer.pad_token}")
# Note: if the model is moved to GPU here by using .cuda(), all of the processes in the node end up
# using the same GPU, instead of each using a different GPU.
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

# print(model)

class GPTModel(pl.LightningModule):
    def __init__(self, model_name, lr, steps_train):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.steps_train = steps_train
        self.model_name = model_name
        # self.batch_idx = 0
        # self.config = transformers.AutoConfig.from_pretrained(model_name)
        # weights_path = huggingface_hub.hf_hub_download(model_name, 'pytorch_model.bin')
        # with accelerate.init_empty_weights():
        #     self.model = transformers.AutoModelForCausalLM.from_config(self.config)
        # self.model.tie_weights()
        # self.model = accelerate.load_checkpoint_and_dispatch(self.model, weights_path, device_map='auto', no_split_module_classes=['GPTJBlock'])

    # Shard model on GPU.
    # https://lightning-transformers.readthedocs.io/en/latest/features/large_model_training.html
    # See also: https://github.com/Lightning-AI/lightning/issues/17043
    # See also: https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    # See also: https://discuss.huggingface.co/t/fine-tuning-t5-with-long-sequence-length-using-activation-checkpointing-with-deepspeed/27236
    def setup(self, stage):
        if not hasattr(self, 'model'):
            # enable_transformers_pretrained_deepspeed_sharding(self)
            # transformers.deepspeed._hf_deepspeed_config_weak_ref = self.dsconfig
            # self.trainer.strategy.config['comms_logger'] = {'enabled': True, 'verbose': True, 'prof_all': True, 'debug': False}
            print(f"DeepSpeed configuration: {self.trainer.strategy.config}")
            self.dsconfig = transformers.deepspeed.HfDeepSpeedConfig(self.trainer.strategy.config)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
            self.model.gradient_checkpointing_enable()

    def forward(self, batch):
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          labels=batch['labels'])
        # print(f"{self.batch_idx} {int(torch.min(batch['input_ids']))} {int(torch.max(batch['input_ids']))} {int(torch.min(batch['attention_mask']))} {int(torch.max(batch['attention_mask']))} {int(torch.min(batch['labels']))} {int(torch.max(batch['labels']))}, {batch['input_ids'].shape[1]} {batch['attention_mask'].shape[1]} {batch['labels'].shape[1]}", flush=True)
        # print(f"{self.batch_idx} done", flush=True)
        # self.batch_idx += 1
        # return out

    def training_step(self, batch):
        out = self(batch)
        self.log_dict({'loss': out.loss, 'global_step': self.trainer.global_step}, sync_dist=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        # if batch_idx % 100 == 0:
        #     print(batch, flush=True)
        self.log('val_loss', self(batch).loss, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        max_value = torch.tensor([[max_length]]).cuda()
        output = [self.model.generate(torch.unsqueeze(p[:l], 0), do_sample=False, max_length=max_length).squeeze() for p, l in zip(batch['input_ids'], batch['prefix_length'])]
        # print(f"Output in predict_step: {output}", flush=True)
        # print(batch['input_ids'][0][:batch['prefix_length']])
        # print(f"Number of EOS tokens: {torch.sum(batch['input_ids'][0][:batch['prefix_length']] == tokenizer.eos_token_id)}")
        # print(batch)
        return [o[l:torch.concat([torch.nonzero(o == tokenizer.eos_token_id), max_value])[0]] for o, l in zip(output, batch['prefix_length'])]

    def configure_optimizers(self):
        # Instead of self.parameters(), self.trainer.model.parameters() must be used with FSDP auto-wrapping.
        # See: https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#auto-wrapping
        # optimizer = transformers.optimization.AdamW(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.01)
        optimizer = deepspeed.ops.adam.FusedAdam(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.01)
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.01)
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
        dataloader = torch.utils.data.DataLoader(torch_dataset, collate_fn=PromptMaskingDataCollator(tokenizer=self.tokenizer, mlm=False), batch_size=self.batch_size, sampler=sampler, num_workers=1, pin_memory=True)
        print(f"Training dataset size: {len(self.dataset['train'])}, dataloader size: {len(dataloader)}")
        return dataloader

    def val_dataloader(self):
        torch_dataset = OCRDataSet(self.dataset['test'])
        # sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(torch_dataset, collate_fn=PromptMaskingDataCollator(tokenizer=self.tokenizer, mlm=False), batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=False)
        print(f"Dataset size: {len(self.dataset['test'])}, dataloader size: {len(dataloader)}")
        return dataloader

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
    # start = time.time()
    cer = evaluate.load('character').compute(predictions=predictions, references=references)
    wer = evaluate.load('wer').compute(predictions=predictions, references=references)
    # print(f"Time to evaluate all at once: {time.time() - start} s")
    return {'cer': cer['cer_score'], 'wer': wer}

def tokenize_with_prefix_length(tokenizer, b):
    prefix = tokenizer(['Input:\n'+i+'\n\nOutput:\n' for i in b['input']], truncation=False)
    output = tokenizer([o + tokenizer.eos_token for o in b['output']], truncation=False)
    d = {**{k: [p + o for p, o in zip(prefix[k], output[k])] for k in prefix}, 'prefix_length': [len(p) for p in prefix['input_ids']]}
    return d

class PredWriter(pl.callbacks.BasePredictionWriter):
    def __init__(self, references, **kwargs):
        super().__init__(**kwargs)
        self.references = references

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_idx):
        print(f"World size: {torch.distributed.get_world_size()}")
        print(f"Number of predictions in PredWriter: {len(predictions)}")
        print(f"Value of batch_idx: {batch_idx}")
        print(len(predictions))
        predictions_gathered = [None]*torch.distributed.get_world_size()
        torch.distributed.all_gather_object(predictions_gathered, predictions)
        batch_idx_gathered = [None]*torch.distributed.get_world_size()
        torch.distributed.all_gather_object(batch_idx_gathered, batch_idx)
        torch.distributed.barrier()
        if not trainer.is_global_zero:
            return
        predictions_gathered = tokenizer.batch_decode([p for b in predictions_gathered for t in b for p in t])
        batch_idx_gathered = [n for b in batch_idx_gathered for t in b for l in t for n in l]
        print(f"Number of gathered predictions in PredWriter: {len(predictions_gathered)}, number of references: {len(self.references)}")
        print(f"Gathered batch_idx: {batch_idx_gathered}")
        predictions_gathered = [p for _, p in sorted(zip(batch_idx_gathered, predictions_gathered))]
        for p, r in zip(predictions_gathered, self.references):
            print([p[:100]])
            print([r[:100]])
            print()

        metrics = compute_metrics(predictions=predictions_gathered, references=self.references)
        cer_scores, wer_scores = [[metric.compute(predictions=[p], references=[r]) for p, r in zip(predictions_gathered, self.references)] for metric in [evaluate.load('character'), evaluate.load('wer')]]
        cer_scores = [s['cer_score'] for s in cer_scores]

        print(f"Average prediction length: {sum(len(s) for s in predictions) / len(predictions)}")
        print(f"Average reference length: {sum(len(s) for s in references) / len(references)}")
        print(f"CER: {metrics['cer']}, WER: {metrics['wer']}, CER (from individual): {sum(cer_scores)/len(cer_scores)}, WER (from individual): {sum(wer_scores)/len(wer_scores)}")
        print(f"Validation size: {len(datamodule.dataset['test'])}, dataloader size: {len(datamodule.val_dataloader())}, number of predictions: {len(predictions_gathered)}, number of references: {len(self.references)}, number of cer scores: {len(cer_scores)}, number of wer scores: {len(wer_scores)}")
        print(f"CER scores: {' '.join(f'{n:.4f}' for n in cer_scores)}")
        print(f"WER scores: {' '.join(f'{n:.4f}' for n in wer_scores)}")
        print(f"CER mean: {torch.mean(torch.tensor(cer_scores))}, stdev: {torch.std(torch.tensor(cer_scores))}")
        print(f"WER mean: {torch.mean(torch.tensor(wer_scores))}, stdev: {torch.std(torch.tensor(wer_scores))}")

        plots = True
        if plots:
            path = pathlib.Path('plots')
            ax1 = sns.histplot(cer_scores, log_scale=False)
            ax1.set_xscale('symlog', linthresh=1)
            ax1.set_xlabel('CER')
            plt.show()
            plt.savefig(path / 'cer_histogram.pdf')
            plt.close()
            ax2 = sns.histplot(wer_scores, log_scale=False)
            ax2.set_xscale('symlog', linthresh=1)
            ax2.set_xlabel('WER')
            plt.show()
            plt.savefig(path / 'wer_histogram.pdf')
            plt.close()

if __name__ == '__main__':
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
    lr = 2e-5
    local_batch_size = 1
    max_length = 2048
    train_size = 500000
    eval_size = 1000

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print(f"Number of nodes: {args.nodes}, number of GPUs per node: {args.gpus}")
    print(f"Model name: {model_name}")
    print(f"Learning rate: {lr}, local batch size: {local_batch_size}, accumulated gradient batches: {accumulate_grad_batches}, maximum sequence length: {max_length}")
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
    # dataset['train'] = dataset['train'].select(range(min(train_size, len(dataset['train']))))
    # print(dataset['test'][0])
    # print(dataset['test'][-1])
    # print(list(zip(dataset['test']['input'][:10], dataset['test']['output'][:10])))
    # metrics = evaluate.combine(['character', 'wer'], force_prefix=True)
    # print(metrics.compute(predictions=dataset['test']['input'], references=dataset['test']['output']))
    print(dataset)
    print(f"Metrics for copying input: {compute_metrics(predictions=dataset['test']['input'], references=dataset['test']['output'])}")
    # dataset = dataset.map(
    #         lambda d: {k: v.replace('ſ', 's') for k, v in d.items()},
    #         num_proc=4
    # )
    # print(dataset)
    # print(dataset['test'][0])
    # print(dataset['test'][-1])

    # stride = 10000
    # for split in ['train', 'test']:    
    #     tokenized_dataset = datasets.Dataset.from_dict(dict())
    #     # This slightly complicated approach is necessary to get the desired number of filtered training examples
    #     # without necessarily having to tokenize the entire dataset.
    #     for i in range(0, len(dataset[split]), stride):
    #         # https://huggingface.co/docs/transformers/tasks/language_modeling
    #         subset = dataset[split][i:max(i+stride, len(dataset[split]))]
    #         subset = subset.map(
    #             lambda b: tokenize_with_prefix_length(tokenizer, b),
    #             batched=True,
    #             num_proc=4
    #         )
    # 
    #         subset = subset.remove_columns(['input', 'output'])
    #         # Inputs longer than the maximum length of the model are removed from the dataset.
    #         subset = filter_by_length(dataset, max_length)
    #         tokenized_dataset = datasets.concatenate_datasets([tokenized_dataset, subset])
    #         if len(tokenized_dataset) >= train_size:
    #             break

    # https://huggingface.co/docs/transformers/tasks/language_modeling
    dataset = dataset.map(
        lambda b: tokenize_with_prefix_length(tokenizer, b),
        batched=True,
        num_proc=4
    )

    dataset = dataset.remove_columns(['input', 'output'])
    # Inputs longer than the maximum length of the model are removed from the dataset.
    dataset = filter_by_length(dataset, max_length)

    print(dataset)
    # print(dataset['test'][:10])

    # truncated = sum(len(e) == max_length for e in dataset['train']['input_ids'] + dataset['test']['input_ids'])
    # print(f"Number of maximum length samples: {truncated}, proportion: {truncated / (len(dataset['train']) + len(dataset['test']))}")

    datamodule = OCRDataModule(dataset, tokenizer, local_batch_size)
    steps_train = math.ceil(train_size / (args.nodes*args.gpus*local_batch_size*accumulate_grad_batches))
    print(f"Number of training steps: {steps_train}", flush=True)

    if args.load_checkpoint:
        # gpt_model = GPTModel.load_from_checkpoint(args.load_checkpoint)
        print(f"Model loaded from checkpoint: {args.load_checkpoint}")

    gpt_model = GPTModel(model_name, lr, steps_train)

    references = []
    for batch in datamodule.val_dataloader():
        references += [o[l:torch.nonzero(o == tokenizer.eos_token_id)[0]] for o, l in zip(batch['input_ids'], batch['prefix_length'])]
    references = tokenizer.batch_decode(references)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=steps_train, monitor='global_step', mode='max', save_top_k=-1, dirpath=args.out_dir, filename=(model_name+'-{global_step}').replace('/', '_'))
    checkpoint_epoch_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_on_train_epoch_end=True, save_top_k=-1, dirpath=args.out_dir, filename=(model_name+'-{epoch}').replace('/', '_'))
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=1, dirpath=args.out_dir, filename=model_name+'-{epoch}')

    logger = pl.loggers.TensorBoardLogger('tb_logs', name=f"{model_name}-{train_size}".replace('/', '_'))

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
        # strategy=pl.strategies.DeepSpeedStrategy(stage=3, offload_optimizer=True, partition_activations=True),
        # strategy=pl.strategies.DeepSpeedStrategy(stage=3, offload_optimizer=True, sub_group_size=1e11, allgather_bucket_size=2e7, reduce_bucket_size=2e7),
        # strategy=pl.strategies.DeepSpeedStrategy(stage=3, sub_group_size=10000000000, allgather_bucket_size=2000000, reduce_bucket_size=2000000),
        strategy='deepspeed_stage_2',
        precision=16,
        # gradient_clip_algorithm='norm',
        # gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=100,
        # limit_val_batches=0.0,
        # max_epochs=1,
        max_steps=steps_train,
        callbacks=[checkpoint_callback, checkpoint_epoch_callback, pl.callbacks.TQDMProgressBar(refresh_rate=10), PredWriter(references=references, write_interval='epoch')],
        logger=logger
    )

    print(trainer.global_rank, trainer.world_size, os.environ['SLURM_NTASKS'])

    trainer.fit(gpt_model, datamodule=datamodule, ckpt_path=args.load_checkpoint)

    # Use predict_step to speed up evaluation.
    print("Evaluating.")
    start = time.time()
    trainer.predict(gpt_model, datamodule.val_dataloader(), ckpt_path=args.load_checkpoint)
    print(f"Time elapsed during evaluation: {time.time() - start} s")
    # print(f"Number of predictions: {len(predictions)}, validation dataset size: {len(datamodule.val_dataloader())}", flush=True)
    # print([len(t) for t in predictions], flush=True)
    # print([len(p) for t in predictions for p in t], flush=True)
    print(f"The default process group of torch.distributed is initialized: {torch.distributed.is_initialized()}")
    print(f"Backend used for torch.distributed: {torch.distributed.get_backend()}")
    print(f"Rank: {torch.distributed.get_rank()}, world size: {torch.distributed.get_world_size()}")
    # print(len(predictions), flush=True)

    # gpt_model.eval()
    # save_dir = f'{args.out_dir}/{model_name.replace("/", "_")}'
    # gpt_model.model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)
