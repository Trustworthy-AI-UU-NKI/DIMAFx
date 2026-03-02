import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from sksurv.metrics import concordance_index_censored

from .losses import NLLSurvLoss, CoxLoss, DisentangledSurvLoss
from .test import test_survival_model
from embeddings.embeddings import prepare_embeddings
from models.DIMAFx import DIMAFx
from utils.general_utils import save_json
from utils.train_utils import get_optim, get_lr_scheduler, list_to_device, LoggingMeter, log_results


def survival_train(args, fold, train_dl, test_dl=None):
    """ Train a survival prediction model for a single fold. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up results and log dir.
    result_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}")
    log_dir_fold = os.path.join(args.log_dir, f"Fold_{fold}")
    results = {}

    os.makedirs(result_dir_fold, exist_ok=True)
    os.makedirs(log_dir_fold, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir_fold)

    # Initialize loss function
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        num_classes = args.n_label_bins
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        num_classes = 1
    else:
        loss_fn_split = args.loss_fn.split("_")
        loss_fn = DisentangledSurvLoss(loss_fn_split[0], loss_fn_split[1], weight_surv=args.w_surv, weight_disentanglement=args.w_dis, n_label_bins=args.n_label_bins, alpha=args.nll_alpha)
        num_classes = loss_fn.get_num_classes()

    print('\nCreate unimodal representations...', end=' ')
    train_dl, data_info  = prepare_embeddings(args, 'train', train_dl)
    
    if not test_dl == None:
        test_dl, _ = prepare_embeddings(args, 'test', test_dl)
    
    ####### WE ARE HERE WITH CHECKING #########
    print('\nInit Model...', end=' ')
    model = DIMAFx(rna_dims=data_info['Pathway sizes'],
                       histo_dim=data_info['Dim wsi'],
                       device=device,
                       single_out_dim=256,
                       num_classes=num_classes,
                       loss_fn=loss_fn,
                       aggr_post_embed=args.aggr_post_embed,
                       wsi_representation_type=args.wsi_repr
                       )
    model.to(device)
    
    print('\nInit optimizer ...')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_dl))

    #####################
    # The training loop #
    #####################
    # Logging
    if not test_dl == None:
        init_results = test_survival_model(model, test_dl, device, return_attn=True, result_dir=result_dir_fold, mode='pre_training')
        log_results(writer, init_results, -1, mode='test')

    for epoch in range(args.max_epochs):
        # Train
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results, train_data_info = train_loop(model, train_dl, optimizer, lr_scheduler, device)
        log_results(writer, train_results, epoch, mode='train')

        # Logging
        if not test_dl == None:
            int_results = test_survival_model(model, test_dl, device, survival_info_train=train_data_info, mode='during_training')
            log_results(writer, int_results, epoch, mode='test')
        
    # Save last model
    torch.save(model.state_dict(), os.path.join(result_dir_fold, "model_checkpoint.pth"))
        

    # End of epoch: Save the last train and test results
    print(f'End of training. Evaluating on Split {fold}...:')
    if not test_dl == None:
        results = test_survival_model(model, test_dl, device, survival_info_train=train_data_info, return_attn=True, result_dir=result_dir_fold)
        save_json(result_dir_fold, f"train_test_summary.json", results)

    writer.close()
    return results

def train_loop(model, dataloader, optimizer, lr_scheduler, device):
    """
        Train loop for survival prediction
    """
    model.train()
    train_log = {}
    all_risk_scores, all_censorships, all_event_times = [], [], []

    # Loop over all data samples
    for idx, batch in enumerate(dataloader):
        # Get the data and labels
        wsi = batch['img'].to(device)
        rna = list_to_device(batch['rna'], device)

        label = batch['label'].to(device)
        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)

        # Forward pass
        output_results, log_dict = model(wsi, rna, label=label, censorship=censorship)

        # Backward pass
        loss = output_results['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # For logging purposes
        for key, val in log_dict.items():
            if key not in train_log:
                train_log[key] = LoggingMeter(key)
            train_log[key].update(val, n=len(wsi))
        
        all_risk_scores.append(output_results['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())
        
    
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    
    # Compute c-index
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    results = {item: meter.avg for item, meter in train_log.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    train_data_info = {'censorship': all_censorships, 'time':all_event_times}
    return results, train_data_info
    