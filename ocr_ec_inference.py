import torch
import transformers
import datasets
import evaluate
import deepspeed
import argparse
import time
import collections
import os
import json
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

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

def filter_by_length(datasetdict, max_length):
    for k in datasetdict:
        filtered = datasetdict[k].filter(lambda e: len(e['input_ids']) <= max_length)
        orig_length = len(datasetdict[k]['input_ids'])
        filt_length = len(filtered['input_ids'])
        print(f'filtered {k} from {orig_length} to {filt_length}')
        print(f'({filt_length/orig_length:.1%}) by max_length {max_length}')
        datasetdict[k] = filtered

    return datasetdict

def tokenize_with_prefix_length(tokenizer, b):
    prefix = tokenizer(['Input:\n'+i+'\n\nOutput:\n' for i in b['input']], truncation=False)
    output = tokenizer([o + tokenizer.eos_token for o in b['output']], truncation=False)
    prefix = {'prefix_' + k: v for k, v in prefix.items()}
    output = {'output_' + k: v for k, v in output.items()}
    d = {**prefix, **output}
    return d

def compute_metrics(predictions, references):
    cer = evaluate.load('character').compute(predictions=predictions, references=references)
    wer = evaluate.load('wer').compute(predictions=predictions, references=references)
    return {'cer': cer['cer_score'], 'wer': wer}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', help="Automatically given by the DeepSpeed launcher.")
    parser.add_argument('--nodes', type=int, default=1, help="Number of nodes.")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use per node.")
    parser.add_argument('--train', help="A jsonl file, with each row containing a noisy text 'input' and its correct form 'output'.")
    parser.add_argument('--eval', help="A jsonl file in the same format as the --train argument.")
    parser.add_argument('--out_dir', help="A directory to which the model checkpoints are saved.")
    parser.add_argument('--predictions', help="A path to which the predictions are saved.")
    parser.add_argument('--references', help="A path to which the references are saved.")
    parser.add_argument('--load_checkpoint', help="A path to a checkpoint file to load.")
    args = parser.parse_args()

    # accumulate_grad_batches = 1
    # lr = 2e-5
    local_batch_size = 8
    max_length = 1000
    # eval_size = 1000

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    dsconfig = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": 'none',
                "pin_memory": True
            },
            "offload_param": {
                "device": 'none',
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e12,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto", 
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        'train_micro_batch_size_per_gpu': local_batch_size
        # 'train_batch_size': world_size * accumulate_grad_batches * local_batch_size
    }

    # hf_dsconfig = transformers.deepspeed.HfDeepSpeedConfig(dsconfig)

    # args.load_checkpoint = 'LumiOpen/Poro-34B'

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # config = transformers.AutoConfig.from_pretrained(args.load_checkpoint)

    # with deepspeed.OnDevice(device='meta', dtype=torch.float32, enabled=True):
    #     model = transformers.AutoModelForCausalLM.from_config(config)

    # checkpoint_files = [f for f in os.scandir(os.path.join(args.load_checkpoint, 'checkpoint')) if os.path.isfile(f)]
    # print(f"Checkpoint files: {checkpoint_files}")

    # checkpoint_dict = {'type': 'DS_MODEL', 'checkpoints': checkpoint_files, 'version': 1.0}

    # model = deepspeed.init_inference(model, max_out_tokens=max_length, dtype=torch.float32, tensor_parallel={'tp_size': world_size}, replace_with_kernel_inject=False, checkpoint=checkpoint_dict)

    # print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory}, reserved: {torch.cuda.memory_reserved(0)}, allocated: {torch.cuda.memory_allocated(0)}")

    model = transformers.AutoModelForCausalLM.from_pretrained(args.load_checkpoint, torch_dtype='auto', device_map='auto')
    # model = transformers.AutoModelForCausalLM.from_pretrained(args.load_checkpoint, torch_dtype='auto')
    # ds_engine = deepspeed.init_inference(model, max_out_tokens=max_length, dtype=torch.float32, tensor_parallel={'tp_size': world_size}, replace_with_kernel_inject=True, quant={'enabled': False})

    # print([k for k in model.state_dict().keys()])

    # ds_engine = deepspeed.initialize(model=model, config_params=dsconfig)[0]
    # model = ds_engine.module
    model.eval()

    # save_dir = f'{args.out_dir}/{args.load_checkpoint.replace("/", "_")}_{time.strftime("%Y-%m-%d")}'
    # model.save_pretrained(save_dir)
    # model.config.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)
    # print(model)
    # print(model.hf_device_map)

    # dataset = datasets.load_dataset('json', data_files={'train': args.train, 'test': args.eval})
    dataset_test = datasets.load_dataset('json', data_files={'test': args.eval}, split='test')
    dataset = datasets.DatasetDict({'test': dataset_test})

    print(f"PyTorch local rank: {local_rank}, world size: {world_size}")
    print(f"Inference batch size: {local_batch_size}")
    print(f"Metrics for copying input: {compute_metrics(predictions=dataset['test']['input'], references=dataset['test']['output'])}")

    # https://huggingface.co/docs/transformers/tasks/language_modeling
    dataset = dataset.map(
        lambda b: tokenize_with_prefix_length(tokenizer, b),
        batched=True,
        num_proc=4
    )

    def batched(l, batch_size):
        batch = []
        for t in l:
            batch.append(t)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    dataset_batched = batched([t for t in dataset['test']], local_batch_size)
    # print(f"Dataset keys before batching: {[[list(d.keys()) for d in b] for b in dataset_batched]}")
    # print(f"Dataset length before batching: {len(dataset_batched)}")
    dataset_batched = ({k: [d[k] for d in b] for k in b[0].keys()} for b in dataset_batched)

    # print(f"Test dataset batched length: {len(dataset_batched)}, keys: {[list(b.keys()) for b in dataset_batched]}, length of values: {[[len(v) for v in b.values()] for b in dataset_batched]}")

    # dataset = dataset.select_columns(['input_ids', 'attention_mask', 'prefix_length'])
    # Inputs longer than the maximum length of the model are removed from the dataset.
    # dataset = filter_by_length(dataset, max_length)
    for k in dataset:
        dataset[k] = dataset[k].add_column('id', list(range(len(dataset[k]))))
    torch_dataset = OCRDataSet(dataset['test'])

    # dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=local_batch_size, num_workers=1, pin_memory=True, shuffle=False)

    metric = evaluate.load('character', 'wer')

    print(f"Batch tokenization example: {tokenizer(['Testisyöte 1: ', 'Testisyöte 2, joka on hieman pidempi: '], padding=True, return_tensors='pt')}")

    print(f"Dataset overview before predictions: {dataset}")
    predictions = []
    references = []
    # batch_indexes = []
    generation_start = time.time()
    for idx, batch in enumerate(dataset_batched):
        # batch = {k: v.to(device=local_rank) for k, v in cpu_batch.items()}
        print(f"Running inference for batch number {idx}.")
        start = time.time()
        # prefixes = [p[:l] for p, l in zip(batch['input_ids'], batch['prefix_length'])]
        max_prefix_length = max(len(p) for p in batch['prefix_input_ids'])
        # paddings = [torch.full([max_prefix_length - len(p)], tokenizer.pad_token_id, dtype=p.dtype, device=p.device) for p in prefixes]
        prefixes = [torch.tensor(p, device=local_rank) for p in batch['prefix_input_ids']]
        prefixes = [torch.cat([torch.full([max_prefix_length - len(p)], tokenizer.pad_token_id, dtype=p.dtype, device=p.device), p]) for p in prefixes]
        attention_mask = torch.stack([p != tokenizer.pad_token_id for p in prefixes])
        with torch.no_grad():
            output = model.generate(torch.stack(prefixes), attention_mask=attention_mask, do_sample=False, max_length=max_length)
        max_value = torch.tensor([[max_length]]).cuda()
        # prefixes += [p[:l] for p, l in zip(batch['input_ids'], batch['prefix_length'])]
        batch_prediction = [o[len(p):] for o, p in zip(output, prefixes)]
        batch_prediction = [o[:torch.concat([torch.nonzero(o == tokenizer.eos_token_id), max_value])[0]] for o in batch_prediction]
        # batch_reference = [i[len(p):] for i, p in zip(batch['output_input_ids'], prefixes)]
        # batch_reference = [o[l:torch.nonzero(o == tokenizer.eos_token_id)[0]] for o, l in zip(batch['input_ids'], batch['prefix_length'])]
        predictions += batch_prediction
        references += batch['output_input_ids']
        # batch_indexes += batch['id']
        end = time.time()
        print(f"Inference for batch number {idx} completed in {end - start} s.", flush=True)

    generation_end = time.time()
    print(f"Prediction generation completed in {generation_end - generation_start} s.")

    # print(f"Predictions: {[(i, p) for i, p in enumerate(predictions)]}")
    # print(f"References: {[(i, r) for i, r in enumerate(references)]}")

    if local_rank == 0:
        prediction_tokens = [p.to(device='cpu') for p in predictions]
        # reference_tokens = [r.to(device='cpu') for r in references]
        predictions = tokenizer.batch_decode(prediction_tokens)
        references = tokenizer.batch_decode(references)
        print(f"Prediction lengths after decoding: {[len(s) for s in predictions]}")
        print(f"Reference lengths after decoding: {[len(s) for s in references]}")
        # batch_indexes = [int(i) for i in batch_indexes]

        for p in predictions[:10]:
            print(p)
        for r in references[:10]:
            print(r)
        # print(f"Batch indexes: {batch_indexes}")

        # Use batch indexes to check that all repeated outputs are identical, and then filter them.
        # prediction_dict = collections.defaultdict(list)
        # for k, v in zip(batch_indexes_gathered, predictions_gathered):
        #     prediction_dict[k].append(v)

        # reference_dict = collections.defaultdict(list)
        # for k, v in zip(batch_indexes_gathered, references_gathered):
        #     reference_dict[k].append(v)

        # # print(f"Prediction dictionary: {prediction_dict}")
        # # print(f"Reference dictionary: {reference_dict}")

        # if not all([all([torch.equal(v[0], p) for v in d.values() for p in v[1:]]) for d in [prediction_dict, reference_dict]]):
        #     print("There are repeated samples where the references and/or predictions do not match. Exiting run.")
        #     raise SystemError

        # metrics = metric.compute()

        # predictions = tokenizer.batch_decode([v[0] for v in prediction_dict.values()])
        # references = tokenizer.batch_decode([v[0] for v in reference_dict.values()])

        for filename, lines in [(args.predictions, predictions), (args.references, references)]:
            with open(filename, 'wt', encoding='utf-8') as f:
                for l in lines:
                    f.write(json.dumps({'output': l}) + '\n')

        with open(args.references, 'wt', encoding='utf-8') as f:
            for r in references:
                f.write(json.dumps({'output': r}) + '\n')

        metrics = compute_metrics(predictions=predictions, references=references)
        cer_scores, wer_scores = [[metric.compute(predictions=[p], references=[r]) for p, r in zip(predictions, references)] for metric in [evaluate.load('character'), evaluate.load('wer')]]
        cer_scores = [s['cer_score'] for s in cer_scores]

        print(f"Number of predictions: {len(predictions)}")
        print(f"Average prediction length: {sum(len(s) for s in predictions) / len(predictions)} tokens")
        print(f"Average reference length: {sum(len(s) for s in references) / len(references)} tokens")
        print(f"CER: {metrics['cer']}, WER: {metrics['wer']}, CER (from individual): {sum(cer_scores)/len(cer_scores)}, WER (from individual): {sum(wer_scores)/len(wer_scores)}")
        print(f"Validation size: {len(dataset['test'])}, number of predictions: {len(predictions)}, number of references: {len(references)}, number of cer scores: {len(cer_scores)}, number of wer scores: {len(wer_scores)}")
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
