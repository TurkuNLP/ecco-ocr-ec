import torch
import transformers
import datasets
import accelerate
import evaluate
import argparse
import time
# import lightning as L

# fabric = L.Fabric(accelerator='cuda', devices=8, strategy='deepspeed_stage_3')
# fabric.launch()

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
    d = {**{k: [p + o for p, o in zip(prefix[k], output[k])] for k in prefix}, 'prefix_length': [len(p) for p in prefix['input_ids']]}
    return d

dsconfig = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": 'none',
            "pin_memory": True
        },
        "offload_param": {
            "device": 'cpu',
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
    'train_micro_batch_size_per_gpu': 1
}

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
    max_length = 128
    # train_size = 2000
    eval_size = 1000

    accelerator = accelerate.Accelerator(deepspeed_plugin=accelerate.DeepSpeedPlugin(hf_ds_config=accelerate.utils.deepspeed.HfDeepSpeedConfig(dsconfig), zero_stage=3))

    # configuration = transformers.AutoConfig.from_pretrained(args.load_checkpoint)
    # with accelerate.init_empty_weights():
    #     model = transformers.AutoModelForCausalLM.from_config(configuration)

    # print("Model loaded with empty weights.")
    # device_map = accelerate.infer_auto_device_map(model, max_memory= {0: '10GiB', **{k: '40GiB' for k in range(1, args.gpus)}})
    # print(device_map)
    # model = accelerate.load_checkpoint_and_dispatch(model, checkpoint=args.load_checkpoint, device_map=device_map)
    # accelerator.print(model.hf_device_map)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory}, reserved: {torch.cuda.memory_reserved(0)}, allocated: {torch.cuda.memory_allocated(0)}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.load_checkpoint)
    model.eval()

    # save_dir = f'{args.out_dir}/{args.load_checkpoint.replace("/", "_")}_{time.strftime("%Y-%m-%d")}'
    # model.save_pretrained(save_dir)
    # model.config.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)
    # print(model)

    # dataset = datasets.load_dataset('json', data_files={'train': args.train, 'test': args.eval})
    dataset = datasets.load_dataset('json', data_files={'test': args.eval})
    dataset['test'] = dataset['test'].select(range(eval_size))
    # https://huggingface.co/docs/transformers/tasks/language_modeling
    dataset = dataset.map(
        lambda b: tokenize_with_prefix_length(tokenizer, b),
        batched=True,
        num_proc=4
    )

    dataset = dataset.remove_columns(['input', 'output'])
    # Inputs longer than the maximum length of the model are removed from the dataset.
    dataset = filter_by_length(dataset, max_length)
    torch_dataset = OCRDataSet(dataset['test'])

    dataloader = torch.utils.data.DataLoader(torch_dataset, collate_fn=PromptMaskingDataCollator(tokenizer=tokenizer, mlm=False), batch_size=local_batch_size, num_workers=1, pin_memory=True, shuffle=False)

    print("Preparing model and dataloader.")
    model, dataloader = accelerator.prepare(model, dataloader)
    # dataloader = accelerator.prepare(dataloader)
    # model = accelerator.prepare(model)

    metric = evaluate.load('character', 'wer')

    # accelerator.print(dataloader)
    # print({p.get_device() for p in model.parameters()})
    # with accelerator.split_between_processes(list(range(len(torch_dataset)))) as subset_idx:
    #     # process_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(torch_dataset, subset_idx), collate_fn=PromptMaskingDataCollator(tokenizer=tokenizer, mlm=False), batch_size=local_batch_size, num_workers=1, pin_memory=True, shuffle=False)
    #     # print(f"Data subset indexes in process {accelerator.process_index}: {subset_idx}, subset: {torch_dataset[subset_idx]}, dataloader length: {len(process_dataloader)}", flush=True)
    #     predictions = []
    #     references = []
    #     for batch in dataloader:
    #         # batch = {k: v.to(device='cuda') for k,v in cpu_batch.items()}
    #         with torch.no_grad():
    #             output = [model.generate(torch.unsqueeze(p[:l], 0), do_sample=False, max_length=max_length).squeeze() for p, l in zip(batch['input_ids'], batch['prefix_length'])]
    #         max_value = torch.tensor([[max_length]]).cuda()
    #         batch_prediction = [o[l:torch.concat([torch.nonzero(o == tokenizer.eos_token_id), max_value])[0]] for o, l in zip(output, batch['prefix_length'])]
    #         batch_reference = [o[l:torch.nonzero(o == tokenizer.eos_token_id)[0]] for o, l in zip(batch['input_ids'], batch['prefix_length'])]

    #         print(batch_prediction)
    #         print(batch_reference)
    #         predictions += batch_prediction
    #         references += batch_reference

    #     predictions_gathered = [None]*torch.distributed.get_world_size()
    #     references_gathered = [None]*torch.distributed.get_world_size()
    #     torch.distributed.all_gather_object(predictions_gathered, predictions)
    #     torch.distributed.all_gather_object(references_gathered, references)
    #     torch.distributed.barrier()

    #     # if main process:
    #     # decode predictions and references with the tokenizer
    #     # compute metrics
    #     # print results
    #     accelerator.print(predictions_gathered)
    #     accelerator.print(references_gathered)
    #     predictions_gathered = tokenizer.batch_decode(predictions_gathered)
    #     references_gathered = tokenizer.batch_decode(references_gathered)
    #     # if accelerator.is_main_process:

    for batch in dataloader:
        with torch.no_grad():
            output = [model.generate(torch.unsqueeze(p[:l], 0), do_sample=False, max_length=max_length).squeeze() for p, l in zip(batch['input_ids'], batch['prefix_length'])]
        max_value = torch.tensor([[max_length]]).cuda()
        # prefixes += [p[:l] for p, l in zip(batch['input_ids'], batch['prefix_length'])]
        batch_prediction = [o[l:torch.concat([torch.nonzero(o == tokenizer.eos_token_id), max_value])[0]] for o, l in zip(output, batch['prefix_length'])]
        batch_reference = [o[l:torch.nonzero(o == tokenizer.eos_token_id)[0]] for o, l in zip(batch['input_ids'], batch['prefix_length'])]
        # predictions = []
        # references = []
        # for prediction, reference in zip(batch_prediction, batch_reference):
        #     accelerator.print(prediction)
        #     accelerator.print(reference)
        #     gathered_prediction, gathered_reference = accelerator.gather_for_metrics((prediction, reference))
        #     predictions += gathered_prediction
        #     references += gathered_reference

        accelerator.print(batch_prediction)
        accelerator.print(batch_reference)
        batch_prediction = tokenizer.batch_decode(batch_prediction)
        batch_reference = tokenizer.batch_decode(batch_reference)
        gathered = accelerator.gather_for_metrics((batch_prediction, batch_reference))
        accelerator.print(f"Value of gathered: {gathered}")
        metric.add_batch(predictions=predictions, references=references)

    metrics = metric.compute()

    # print(f"Metrics for copying input: {compute_metrics(predictions=dataset['test']['input'], references=dataset['test']['output'])}")
    accelerator.print(metrics)
