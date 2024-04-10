from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from transformers import  AdamW, get_linear_schedule_with_warmup
from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.prompts import MixedTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import KnowledgeableVerbalizer
from transformers import get_linear_schedule_with_warmup
from openprompt.data_utils.data_sampler import FewShotSampler
from tqdm import tqdm
from openprompt.utils.calibrate import calibrate
import random

import torch
from src.dataDeal.phish_data import *

logger = {}
phish_set = PhishDataset('./data/answer2.xlsx','./data/spam_email_data.log','.')
samples = []
for (i,email) in enumerate(phish_set.samples):
    samples.append(InputExample(
        guid = i,
        label = email.label,
        text_a = email.content,
        text_b = email.subject,
        # todo multi dimension
    ))
logger["dimension"] = 2
random.shuffle(samples)
samples = samples[:len(samples) * 0.1] # todo
logger["sample length"] = len(samples)

train_samples , test_samples,dev_samples = samples[:int(len(samples) * 0.8)] , samples[int(len(samples) * 0.8) :int(len(samples) * 0.9)], samples[int(len(samples) * 0.9) :]
dataset = {}
dataset['train'] = train_samples
dataset['validation'] = dev_samples
dataset['test'] = test_samples

def not_freeze(model_path, plm_freeze = False):
    logger={}
    model_name = model_path.split("-")[0]

    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)


    template_text = 'Content: {"placeholder":"text_a"} Subject: {"placeholder":"text_b"}. This is {"mask"} email.'

    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])

    wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

    tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)

    model_inputs = {}
    for split in ['train', 'validation', 'test']:
        model_inputs[split] = []
        for sample in dataset[split]:
            tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
            model_inputs[split].append(tokenized_example)

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                            label_words=[["phishing"], ["normal"]])

    use_cuda = torch.cuda.is_available()
    print("GPU enabled? {}".format(use_cuda))
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=plm_freeze)


    if use_cuda:
        prompt_model=  prompt_model.cuda()

    # Now the training is standard

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    for epoch in range(5):
        # todo parameters
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 ==1:
                print("hard template: Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)


    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())


    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)

    logger['not-freeze-hard-template-acc'] = f"{acc:.5f}"



    # using mixed template
    plm, tokenizer, model_config, WrapperClass = load_plm(model_path.split("-")[0], model_path)

    mixed_template = '{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"}? This is {"mask"} email.'

    mytemplate_soft1 = MixedTemplate(model=plm, tokenizer=tokenizer, text=mixed_template)
    wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")



    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate_soft1, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    prompt_model = PromptForClassification(plm=plm,template=mytemplate_soft1, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model=  prompt_model.cuda()

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters
    optimizer_grouped_parameters2 = [
        {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)

    for epoch in range(5):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            if step %100 ==1:
                print("mixed template:Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate_soft1, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)

    logger["model-name"] = model_name
    logger["plm-freeze"] = plm_freeze
    logger["template"] = template_text
    logger["mixed-template"] = mixed_template
    logger["mixed-template-acc"] = f"soft template acc is {acc:.5f}"

    with open("./log", "a+") as file:
        file.write('\n' + str(logger))

def freeze(model_path, plm_freeze=True):
    logger = {}
    model_name = model_path.split("-")[0]
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)

    mytemplate_soft2 = MixedTemplate(model = plm, tokenizer = tokenizer,
                                     text = '{"placeholder":"text_a"} {"soft": "quenstion", "duplicate": 50} {"placeholder":"text_b"} {"soft": "yes", "duplicate": 16} {"soft": "no", "duplicate":16} {"soft": "maybe" , "duplicate": 16} {"mask"}.')
    wrapped_t5tokenizer = WrapperClass(max_seq_length = 128, decoder_max_length = 3, tokenizer = tokenizer,
                                       truncate_method = "head")

    myverbalizer = ManualVerbalizer(tokenizer, num_classes = 2,
                                    label_words = [["phishing"], ["normal"]])

    train_dataloader = PromptDataLoader(dataset = dataset["train"], template = mytemplate_soft2, tokenizer = tokenizer,
                                        tokenizer_wrapper_class = WrapperClass, max_seq_length = 256,
                                        decoder_max_length = 3,
                                        batch_size = 4, shuffle = True, teacher_forcing = False,
                                        predict_eos_token = False,
                                        truncate_method = "head")

    use_cuda = True
    ## Freeze the plm
    prompt_model = PromptForClassification(plm = plm, template = mytemplate_soft2, verbalizer = myverbalizer,
                                           freeze_plm = plm_freeze)
    if use_cuda :
        prompt_model = prompt_model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']

    # optimizer_grouped_parameters1 = [
    #     {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # Using different optimizer for prompt parameters and model parameters
    optimizer_grouped_parameters2 = [
        {'params' : [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    ]

    # optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr = 0.3)

    for epoch in range(20) :  # Longer epochs are needed
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader) :
            if use_cuda :
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(prompt_model.template.parameters(), 1.0)
            # optimizer1.step()
            # optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            if step % 100 == 1 :
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush = True)

    validation_dataloader = PromptDataLoader(dataset = dataset["validation"], template = mytemplate_soft2,
                                             tokenizer = tokenizer,
                                             tokenizer_wrapper_class = WrapperClass, max_seq_length = 256,
                                             decoder_max_length = 3,
                                             batch_size = 4, shuffle = False, teacher_forcing = False,
                                             predict_eos_token = False,
                                             truncate_method = "head")

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader) :
        if use_cuda :
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim = -1).cpu().tolist())

    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    logger["model-name"] = model_name
    logger["plm-freeze"] = plm_freeze
    logger["soft-template"] = mytemplate_soft2
    logger["soft-template-acc"] = f"soft template acc is {acc:.5f}"
    logger['freeze-soft-template-acc'] = f"{acc:.5f}"

    with open("./log", "a+") as file :
        file.write('\n' + str(logger))


# kpt for zero shot text classification
# few shot and zero better use a larger model like roberta-large
def few_shot(model_path):
    logger = {}
    model_name = model_path.split("-")[0]
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    label_words = ["bill, 发票", "色情, sex, 赌博", "gamble, 广告", "ad", "培训, 红包, 报名", "投稿,约会"]
    with open("./kpt_label_words.txt", 'w') as fout :
        for ws in label_words :
            fout.write(ws + "\n")
    myverbalizer = KnowledgeableVerbalizer(tokenizer, num_classes = 6).from_file("./kpt_label_words.txt")
    # todo enrich the labels
    mytemplate = ManualTemplate(tokenizer = tokenizer,
                                text = """A {"mask"} email : {"placeholder": "text_a"} {"placeholder": "text_b"}""")
    support_sampler = FewShotSampler(num_examples_total = 200, also_sample_dev = False)
    dataset['support'] = support_sampler(dataset['train'], seed = 1)
    for example in dataset['support'] :
        example.label = -1  # remove the labels of support set for classification
    support_dataloader = PromptDataLoader(dataset = dataset["support"], template = mytemplate, tokenizer = tokenizer,
                                          tokenizer_wrapper_class = WrapperClass, max_seq_length = 128,
                                          batch_size = 4, shuffle = False, teacher_forcing = False,
                                          predict_eos_token = False,
                                          truncate_method = "tail")
    use_cuda = True
    prompt_model = PromptForClassification(plm = plm, template = mytemplate, verbalizer = myverbalizer,
                                           freeze_plm = False)
    if use_cuda :
        prompt_model = prompt_model.cuda()
    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(4)]

    # calculate the calibration logits
    cc_logits = calibrate(prompt_model, support_dataloader)
    print("the calibration logits is", cc_logits)

    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.
    prompt_model.verbalizer.register_calibrate_logits(cc_logits)
    new_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(4)]
    print("Original number of label words per class: {} \n After filtering, number of label words per class: {}".format(
        org_label_words_num, new_label_words_num))

    logger["model-name"] = model_name
    logger["plm-freeze"] = False
    logger["sample"] = "few-shot"

    with open("./log", "a+") as file :
        file.write('\n' + str(logger))




def zero_shot(model_path,):
    label_words = ["bill, 发票", "色情, sex, 赌博", "gamble, 广告", "ad", "培训, 红包, 报名", "投稿,约会"]
    with open("./kpt_label_words.txt", 'w') as fout :
        for ws in label_words :
            fout.write(ws + "\n")
    logger = {}
    print("executing zero shot:")
    model_name = model_path.split("-")[0]
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    mytemplate = ManualTemplate(tokenizer = tokenizer,
                                text = """Content :{"placeholder": "text_a"},Subject : {"placeholder": "text_b"}.This is {"mask"} email""")
    test_dataloader = PromptDataLoader(dataset = dataset["test"], template = mytemplate, tokenizer = tokenizer,
                                       tokenizer_wrapper_class = WrapperClass, max_seq_length = 128,
                                       batch_size = 4, shuffle = False, teacher_forcing = False,
                                       predict_eos_token = False,
                                       truncate_method = "tail")

    myverbalizer = KnowledgeableVerbalizer(tokenizer, num_classes = 6).from_file("./kpt_label_words.txt")
    prompt_model = PromptForClassification(plm = plm, template = mytemplate, verbalizer = myverbalizer,
                                           freeze_plm = False)

    allpreds = []
    alllabels = []
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    pbar = tqdm(test_dataloader)
    use_cuda = torch.cuda.is_available()
    for step, inputs in enumerate(pbar) :
        if use_cuda :
            inputs = inputs.cuda()
            prompt_model.to("cuda:0")
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim = -1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)

    print("test:", acc)  # roughly ~0.853 when using template 0

    logger["model-name"] = model_name
    logger["plm-freeze"] = False
    logger["template"] = mytemplate
    logger["acc"] = (f"acc is {acc:.5f}")
    logger["sample"] = "zero-shot"
    logger['zero-shot-hard-template-acc'] = f"{acc:.5f}"

    with open("./log", "a+") as file :
        file.write('\n' + str(logger))



if __name__ == '__main__':
    # optimize_test("bert-base-chinese")
    zero_shot("bert-base-chinese")

